'''
Author: rh hengrao02@outlook.com
Date: 2024-03-10 09:15:05
LastEditors: rh hengrao02@outlook.com
LastEditTime: 2024-03-10 09:28:16
Description: dataset
'''
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import threading
import random

from scipy.spatial import cKDTree

class FileDataSet(Dataset):
    file_paths: Tuple[str]
    inputs: Tensor
    targets: Tensor

    # after each train_loader shuffle, only the index is shuffled, the data(input and target) is fixed
    def __init__(self, *file_paths: str, transform, device, flag='train',
                 norm_xs=None, OnlyLoad=False, **kwargs):
        assert flag in ['train', 'test', 'valid']
        self.dataset_flag = flag
        self.file_paths = file_paths
        self.transform = transform
        self.device = device
        self._inputs = []
        self._targets = []

        self.NOSCI_Count = 0
        self.OSCI_Count = 0
        self.eps = 1e-6

        self.norm_xs = norm_xs
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                line = f.readline()
                while line:
                    _input, target = self.transform(line, **kwargs)
                    # assert target[0] >= 0.0, 'Error must big than 0.0 it\'s {}'.format(target[0])
                    if _input is not None and target is not None:
                        self._inputs.append(_input)
                        self._targets.append(target)

                    if target[0] < self.eps:
                        self.NOSCI_Count += 1
                    else:
                        self.OSCI_Count += 1
                    line = f.readline()

        if not OnlyLoad:
            self.init_inputs_targets()

    def init_inputs_targets(self):
        self.NOSCI_Count = self.OSCI_Count = 0
        cur_inputs = np.zeros(np.array(self._inputs).shape)
        for i in range(len(cur_inputs)):
            if self._targets[i][0] < self.eps:
                self.NOSCI_Count += 1
            else:
                self.OSCI_Count += 1

            for j in range(len(cur_inputs[i])):
                if self.norm_xs:
                    cur_inputs[i][j] = \
                        (self._inputs[i][j] - self.norm_xs[j][0]) / (self.norm_xs[j][1] - self.norm_xs[j][0])
                else:
                    cur_inputs[i][j] = self._inputs[i][j]

        self.inputs = torch.tensor(cur_inputs, dtype=torch.float64, device=self.device)
        self.targets = torch.tensor(self._targets, dtype=torch.float64, device=self.device)

        self.initial_indices = list(range(self.targets.size(0)))  # all data index in the initial state
        self.initial_prob = np.ones(self.targets.size(0)) / self.targets.size(0)

        self.training_indices = self.initial_indices

    def __getitem__(self, index) -> T_co:
        index = self.training_indices[index]
        inputs = self.inputs[index]
        targets = self.targets[index]
        return inputs, targets, index

    def __len__(self):
        return len(self.training_indices)

    def update_train_indices(self, new_indices=None):
        if new_indices is None:
            self.training_indices = self.initial_indices
        else:
            self.training_indices = new_indices

    def residule_new_data(self, new_inputs, new_targets):
        self._inputs += new_inputs
        self._targets += new_targets

        self.init_inputs_targets()

class Only_Boundary_Initialize_Dataset(FileDataSet):
    file_paths: Tuple[str]
    inputs: Tensor
    targets: Tensor

    # after each train_loader shuffle, only the index is shuffled, the data(input and target) is fixed
    def __init__(self, *file_paths: str, transform, device, flag='train',
                 norm_xs=None, sudo_boundary_ratio=0.05, boundary_K=10, initial_super=True, 
                 fitness_strategy='C', **kwargs):
        super().__init__(*file_paths, transform=transform, device=device, flag=flag,
                         norm_xs=norm_xs, OnlyLoad=True, **kwargs)
        # fitness_strategy: 'C': Combine smilarity with density, 'D': density, 'S': smilarity 
        from scipy.spatial import cKDTree
        self.fitness_strategy = fitness_strategy

        if initial_super:
            self.kd_tree = cKDTree(self._inputs)
            boundary_indices, left_indices = self.initial_boundary_samples(sudo_boundary_ratio, boundary_K)
            # uniform_indices = []

            new_inputs = [self._inputs[i] for i in np.concatenate([boundary_indices])]
            new_targets = [self._targets[i] for i in np.concatenate([boundary_indices])]

            self.left_inputs = [self._inputs[i] for i in np.concatenate([left_indices])]
            self.left_targets = [self._targets[i] for i in np.concatenate([left_indices])]
            self._inputs = new_inputs
            self._targets = new_targets

            self.init_inputs_targets()

    def find_top_k_neighbors(self, point, k=10):
        _, indices = self.kd_tree.query(point, k=k + 1)  # k+1 to exclude the point itself
        return indices[1:]  # Exclude the first index (the point itself)

    def simility_xy(self, point_x, point_y, label_x, label_y):
        dis_label = 0
        dis_truth = 0

        for i in range(len(point_x)):
            dis_truth += (point_y[i] - point_x[i])**2
        dis_truth = np.sqrt(dis_truth)

        for i in range(len(label_x)):
            dis_label += (label_y[i] - label_x[i]) ** 2
        dis_label = np.sqrt(dis_label)

        return dis_label / dis_truth

    def _samples_fitness(self, boundary_K):
        samples_fitness = np.zeros(len(self._inputs))
        samples_smi = np.zeros(len(self._inputs))

        for i in range(len(self._inputs)):
            indices = self.find_top_k_neighbors(self._inputs[i], boundary_K)

            sample_fitness = 0
            for j in indices:
                sample_fitness += self.simility_xy(self._inputs[i], self._inputs[j],
                                                self._targets[i], self._targets[j])

            samples_smi[i] = sample_fitness
            samples_fitness[i] = samples_smi[i]
        
        return samples_fitness

    def initial_boundary_samples(self, sudo_boundary_ratio, boundary_K):
        self.samples_fitness = self._samples_fitness(boundary_K)

        sorted_indices = np.argsort(self.samples_fitness)
        top_percent_count = int(len(self.samples_fitness) * sudo_boundary_ratio)

        return sorted_indices[-top_percent_count:], sorted_indices[:-top_percent_count]
