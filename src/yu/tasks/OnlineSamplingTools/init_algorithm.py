from abc import ABC, abstractmethod
import numpy as np
from sampler import get_truth_line
from typing import List, Any


import multiprocessing
import functools

import sys
sys.path.extend(['../../../../src'])
from yu.tasks.pde_models import transform

class General_Algorithm():

    def __init__(self, type_name, config_root, ode_model_name):
        self.type_name = type_name
        self.config_root = config_root
        self.ode_model_name = ode_model_name

    @abstractmethod
    def load_config(self):
        pass


def init_shared_array(shape, dtype, a):
    import ctypes

    if dtype == np.double:
        c_type = ctypes.c_double
    elif dtype == int:
        c_type = ctypes.c_int32
    elif dtype == np.float:
        c_type = ctypes.c_float
    else:
        c_type = ctypes.c_double

    if type(a) is not np:
        a = np.array(a.copy())
    shared_array_base = multiprocessing.Array(c_type, a.flatten())
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    return shared_array

class Uniform_Algorithm():

    def __init__(self, xs_range, xs_param, param_selected, max_process = 1, ode_model_name: str='HSECC') -> None:
        self.xs_range = xs_range
        self.xs_param = xs_param
        self.param_selected = param_selected

        self.ode_model_name = ode_model_name

        self.max_process = max_process
    
    def produce(self, xs: List[float], shared_arrays):

        xs_selected = shared_arrays['xs_selected']
        ys_selected = shared_arrays['ys_selected']
        ys_weight = shared_arrays['ys_weight']

        new_xs = self.xs_param.copy()
        idx = xs[0]
        xs = xs[1:]
        for i in range(len(self.param_selected)):
            new_xs[self.param_selected[i]] = xs[i]
        truth_line = get_truth_line(new_xs, self.ode_model_name)

        if truth_line is None:
            return
        xs, ys = transform(truth_line, xs_selected, ys_selected, ys_weight, None, 
                               model_name=self.ode_model_name)

        inputs_targets = [idx] + xs + ys

        self.inputs_targets_q.put(inputs_targets)


    def sample(self, num_samples, 
               xs_selected: List[int], ys_selected: List[int],
                ys_weight: List[float] = None,xs_weight: List[List[float]] = None,
                input_xs: List[List[float]] = None):

        if input_xs:
            samples = np.array(input_xs)
            inputs = np.array(samples)
        else:
            samples = []
            for _ in range(len(self.xs_range)):
                low, high = self.xs_range[_]
                samples.append(np.random.uniform(low, high, num_samples))
        
            inputs = np.array(samples).T

        manager = multiprocessing.Manager()
        shared_arrays = manager.dict()

        shared_arrays['xs_selected'] = init_shared_array(np.array(xs_selected).shape, int, xs_selected)
        shared_arrays['ys_selected'] = init_shared_array(np.array(ys_selected).shape, int, ys_selected)
        shared_arrays['ys_weight'] = init_shared_array(np.array(ys_weight).shape, np.double, ys_weight)

        partial_task_function = functools.partial(self.produce, shared_arrays=shared_arrays)


        task_xs_list = [[i] + inputs[i].tolist() for i in range(len(inputs))]

        pool = multiprocessing.Pool(self.max_process)
        self.inputs_targets_q = manager.Queue()  # [[xs, ys],...]
        pool.map(partial_task_function, task_xs_list)
        pool.close()
        pool.join()

        all_data = []
        new_inputs = []
        new_targets = []
        while not self.inputs_targets_q.empty():
            inputs_targets = self.inputs_targets_q.get()
            all_data.append(inputs_targets)
        
        all_data.sort(key=lambda x: x[0])
        for inputs_targets in all_data:
            new_inputs.append(inputs_targets[1:len(xs_selected) + 1])
            new_targets.append(inputs_targets[-len(ys_selected):])

        return new_inputs, new_targets

    def generate_samples(self, num_samples,
                xs_selected: List[int], ys_selected: List[int],
                ys_weight: List[float] = None,xs_weight: List[List[float]] = None):
        return self.sample(num_samples, xs_selected, ys_selected, ys_weight, xs_weight)


class Gene_Alg(General_Algorithm):
    def __init__(self, type_name, config_root, xs_param, xs_weight, param_selected, ode_model_name: str='HSECC'):
        super().__init__(type_name, config_root, ode_model_name)
        self.CrossOver = self.config_root[1][0] # crossOver iter
        self.Mutation = self.config_root[1][1]  # mutation iter

        self.Large_ratio = 0.1
        self.Small_ratio = 0.1
        self.middle_ratio = 0.8

        self.xs_range = xs_weight
        self.param_selected = param_selected
        self.params = xs_param

    def generate_samples(self, inputs, weights,
                         xs_selected: List[int], ys_selected: List[int],
                         ys_weight: List[float] = None,xs_weight: List[List[float]] = None):
        pass