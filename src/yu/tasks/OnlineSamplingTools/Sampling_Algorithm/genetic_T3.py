import sys
sys.path.extend(['../../../../../'])
from yu.core.logger import logger
import torch
import os
import wandb
import numpy as np

from yu.tasks.OnlineSamplingTools.sampler import get_truth_line
from yu.tasks.pde_models import transform

from yu.tasks.OnlineSamplingTools.init_algorithm import Gene_Alg, init_shared_array
import multiprocessing, functools
from typing import List

class Gene_T3_Thread(Gene_Alg):
    def __init__(self, type_name, config_root, xs_param, xs_weight, param_selected,
                 max_threads_at_once=32, ode_model_name:str='HSECC'):
        # ['Gene_T3_Thread', [T, M], ['A', 'D']]
        super().__init__(type_name, config_root, xs_param, xs_weight, param_selected, ode_model_name)

        self.max_threads_at_once = max_threads_at_once

        self.Stage_list = self.config_root[-1]

    def get_stage_xs(self, task_name, inputs, MR, LR, Middle_indices, Large_indices):
        if task_name == 'A':
            # L & L
            idx = np.random.choice(LR, size=2, replace=False)
            id1 = Large_indices[idx[0]]
            id2 = Large_indices[idx[1]]

            xs = []
            for i in range(len(self.param_selected)):
                _x1 = inputs[id1][i]
                _x2 = inputs[id2][i]
                x = np.random.uniform(min(_x1, _x2),
                                        max(_x1, _x2))
                xs.append(x)
        elif task_name == 'D':
            # L & M
            id1 = Large_indices[np.random.choice(LR, size=1, replace=False)[0]]
            id2 = Middle_indices[np.random.choice(MR, size=1, replace=False)[0]]

            xs = []
            for i in range(len(self.param_selected)):
                _x1 = inputs[id1][i]
                _x2 = inputs[id2][i]
                x = np.random.uniform(min(_x1, _x2),
                                        max(_x1, _x2))
                xs.append(x)

        return xs
        
    def get_truth_stage(self, ids, xs, xs_selected, ys_selected, ys_weight, xs_weight):
        new_xs = self.params.copy()
        for i in range(len(self.param_selected)):
            new_xs[self.param_selected[i]] = xs[i]
        truth_line = get_truth_line(new_xs, self.ode_model_name)

        if truth_line is None:
            return
        xs, ys = transform(truth_line, xs_selected, ys_selected, ys_weight, xs_weight, 
                               model_name=self.ode_model_name)

        inputs_targets = [ids] + xs + ys

        self.inputs_targets_q.put(inputs_targets)

    def task_run(self, xs, shared_arrays):
        ids = xs[0]
        xs = xs[1]
        sd = shared_arrays
        self.get_truth_stage(ids, xs, sd['xs_selected'],
                            sd['ys_selected'], sd['ys_weight'], None)

    def stratified_indices(self, weights, Gaussian_Mixture: bool):
        if not Gaussian_Mixture:
            sorted_indices = np.argsort(weights)
            Large_indices = sorted_indices[-int(len(weights) * self.Large_ratio):]
            Small_indices = sorted_indices[:int(len(weights) * self.Small_ratio)]
            Middle_indices = np.setdiff1d(np.arange(len(sorted_indices)), np.union1d(Large_indices, Small_indices))
        else:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=3).fit(weights.reshape(-1, 1))

            relative_three_stage_ids = np.argsort(gmm.means_[:, 0])
            predict = gmm.predict(weights.reshape(-1, 1))

            Small_indices = (predict == relative_three_stage_ids[0]).nonzero()[0].tolist()
            Middle_indices = (predict == relative_three_stage_ids[1]).nonzero()[0].tolist()
            Large_indices = (predict == relative_three_stage_ids[2]).nonzero()[0].tolist()

            if len(Large_indices) < 2 or len(Middle_indices) < 1:
                return self.stratified_indices(weights, False)

        return Large_indices, Small_indices, Middle_indices

    def generate_samples(self, inputs, weights,
                         xs_selected: List[int], ys_selected: List[int],
                         ys_weight: List[float] = None, xs_weight: List[List[float]] = None,
                         Gaussian_Mixture: bool=False):
        weights = weights / np.sum(weights)
        Large_indices, Small_indices, Middle_indices = self.stratified_indices(weights, Gaussian_Mixture)

        LR = list(range(len(Large_indices)))
        SR = list(range(len(Small_indices)))
        MR = list(range(len(Middle_indices)))

        manager = multiprocessing.Manager()
        shared_arrays = manager.dict()

        shared_arrays['xs_selected'] = init_shared_array(np.array(xs_selected).shape, int, xs_selected)
        shared_arrays['ys_selected'] = init_shared_array(np.array(ys_selected).shape, int, ys_selected)
        shared_arrays['ys_weight'] = init_shared_array(np.array(ys_weight).shape, np.double, ys_weight)

        partial_task_function = functools.partial(self.task_run, shared_arrays=shared_arrays)

        new_inputs = []
        new_targets = []

        task_name_list = [self.Stage_list[0]] * self.CrossOver + [self.Stage_list[1]] * self.Mutation
        task_xs_list = []
        for task_name_idx in range(len(task_name_list)):
            task_xs_list.append([task_name_idx, self.get_stage_xs(task_name_list[task_name_idx], inputs, MR, LR, Middle_indices, Large_indices)])

        pool = multiprocessing.Pool(self.max_threads_at_once)
        self.inputs_targets_q = manager.Queue()  # [[xs, ys],...]
        pool.map(partial_task_function, task_xs_list)
        pool.close()
        pool.join()

        ans_list = []
        while not self.inputs_targets_q.empty():
            inputs_targets = self.inputs_targets_q.get()
            ans_list.append(inputs_targets)
        
        ans_list = sorted(ans_list, key=lambda x: x[0])
        for inputs_targets in ans_list:
            new_inputs.append(inputs_targets[1:len(xs_selected) + 1])
            new_targets.append(inputs_targets[-len(ys_selected):])

        return new_inputs, new_targets
