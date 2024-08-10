import os
import sys

sys.path.extend(['../../../../../src'])  # noqa
from pydantic import validator
from yu.core.config import BaseConfig
from yu.tools.time_util import TimeUtil
from typing import List, Any, Callable
from torch import nn as nn

class TaskConfig(BaseConfig):
    seed: int
    gpu_idx: str

    log_dir: str
    tensorboard_prefix: str = None
    train_paths: List[str]  # train data path
    test_paths: List[str]   # test data path

    ode_model_name: str # biological system
    xs_param: List[float]   # default coefficients values (lambda) 
    xs_lb_ub: List[float]   # coefficients range [xs_lb_ub[0] * lambda ~ xs_lb_ub[1] * lambda]
    param_selected: List[int]   # selected coefficients from xs_param

    xs_selected: List[int]  # selected training input indices (0~dimension D-1)
    ys_selected: List[int]  # selected training target indices (0~dimension D'-1)
    ys_default: List[str]   # useless
    xs_weight: List[List[float]]    # useless
    ys_weight: List[float]  # useless
    ys_name: List[str]  # namely y (useless)

    batch_size: int
    base_lr: float

    epoch_n: int    # continue learning epoch
    last_epoch_n: int   # first training epoch
    nn_layers: List[int]    # MLP layers 
    nn_norm: List[str]
    # optimizer
    weight_decay: float
    # warm up
    warm_up_epoch: int
    warm_up_lr_rate: float
    # learning rate alpha
    lr_alpha: float = 0.2
    # only support nn loss function
    loss_func_type: List[Any]
    loss_func: Callable = None
    init_strategy: str = 'kaiming'

    dropout: float = -1
    # Online Sampling Strategy
    iter_count: int = 10
    Algorithm_type: List[Any] = ['Uniform', 32, 500, 250]  # dynamic sampling
    wandb_swith: bool = True    # log in wandb?

    # IS (Finetune/Continue Learning)
    finetune_epoch: int = 200
    finetune_count: int = 1
    tau: float = 0.5

    # Dataset setting
    dataset_type: str = 'default'
    # used for boundary-Only
    uniform_sampling_ratio: float = 0.05
    boundary_sampling_ratio: float = 0.05
    boundary_KNN: int = 10

    # training-strategy
    train_strategy: str # sampling methods
    fitness_strategy: str   # fitness (gradient-based)
    # reload-model
    pretrained_model_path: str = None
    total_training_samples: int = 5000  # total training samples

    Early_Stopping: bool = False

    # Gradient-based Sampling
    Ada_Gradient_Settings: List[Any] = ["None", 5]  # labeled Samples path, iters count
    Gaussian_Mixture: bool = False
    sampling_core: int=32

    exclude_fields: List[str] = [
        'exclude_fields', 'loss_func',
    ]

    @validator('log_dir', pre=True)
    def set_log_dir(cls, v, values):
        name = values.get('name')
        now = values.get('now')
        if name:
            return os.path.join(v, name, TimeUtil.strftime(now, '%Y%m%d_%H%M%S'))
        else:
            return os.path.join(v, TimeUtil.strftime(now, '%Y%m%d_%H%M%S'))

    @validator('tensorboard_prefix')
    def set_tensorboard_prefix(cls, v, values):
        now = values.get('now')
        if not v:
            v = values.get('name')
        if v:
            return f'[{v}][{TimeUtil.strftime(now, "%Y-%m-%d %H:%M:%S")}] '
        else:
            return f'[{TimeUtil.strftime(now, "%Y-%m-%d %H:%M:%S")}] '

    @validator('xs_weight', pre=True)
    def set_xs_weight(cls, v, values):
        # print(1234)
        xs_param = values.get('xs_param')
        xs_lb_ub = values.get('xs_lb_ub')
        param_selected = values.get('param_selected')

        weight = []
        for i in range(len(param_selected)):
            weight.append([xs_param[param_selected[i]] * xs_lb_ub[0], xs_param[param_selected[i]] * xs_lb_ub[1]])
        return weight

    @validator('loss_func', pre=True)
    def set_loss_func(cls, v, values):
        # print(123)
        config = values.get('loss_func_type')
        if hasattr(nn, config[0]):
            loss_func = getattr(nn, config[0])
        return loss_func(*config[1:])

    @validator('Ada_Gradient_Settings', pre=True)
    def set_Ada_Gradient_Settings(cls, v, values):
        save_dir = values.get('save_dir')
        v[0] = os.path.join(save_dir, 'unfiltered_data.csv')
        return v

    @validator('nn_layers', pre=True)
    def set_nn_layers(cls, v, values):
        xs_selected = values.get('xs_selected')
        ys_selected = values.get('ys_selected')
        return [len(xs_selected)] + v + [len(ys_selected)]

    def lr(self, epoch: int):
        if self.pretrained_model_path is not None and self.pretrained_model_path != 'None':
            self.last_epoch_n = 0
            return None
        if epoch <= self.warm_up_epoch:  # line
            return 1e-8 + (1 - self.warm_up_lr_rate) / self.warm_up_epoch * epoch
        return 1 / (self.lr_alpha * (epoch - self.warm_up_epoch) / (self.base_lr * self.last_epoch_n) + 1)
