"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-04-12 11:10
@description:
"""
import os
import random
import sys
from typing import List, Any, Callable

import numpy as np
from pydantic import validator

sys.path.extend(['../../../../src'])  # noqa

from yu.tools.time_util import TimeUtil

import torch
from yu.tasks.pde_models import transform
from yu.nn.model import MLPModel
from yu.core.config import BaseConfig, auto_config
from dataloader import FileDataSet, Only_Boundary_Initialize_Dataset
from torch import optim, nn
from yu.core.logger import logger
from yu.tools.misc import makedir
from torch.optim.lr_scheduler import LambdaLR
import wandb

os.environ["WANDB_API_KEY"] = 'dc41b090309d1e30cbaee9e2af0db706e87fc5ca'
os.environ["WANDB_MODE"] = "offline"


class TaskConfig(BaseConfig):
    seed: int
    gpu_idx: str

    log_dir: str
    tensorboard_prefix: str = None
    train_paths: List[str]
    test_paths: List[str]

    ode_model_name: str
    xs_param: List[float]
    xs_lb_ub: List[float]
    param_selected: List[int]

    xs_selected: List[int]
    ys_selected: List[int]
    ys_default: List[str]
    xs_weight: List[List[float]]
    ys_weight: List[float]
    ys_name: List[str]

    batch_size: int
    base_lr: float

    epoch_n: int
    last_epoch_n: int
    nn_layers: List[int]
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
    Algorithm_type: List[Any] = ['Uniform', 4, 500, 250]    # Sampling Method, core, total samples, filtered samples
    # wandb switch
    wandb_swith: bool = True

    # Point-GN(Finetune)
    finetune_epoch: int = 200
    finetune_count: int = 1
    tau: float = 0.5

    # Dataset-Boundary
    dataset_type: str = 'default'
    uniform_sampling_ratio: float = 0.05
    boundary_sampling_ratio: float = 0.05
    boundary_KNN: int = 10

    # training-strategy
    train_strategy: str
    fitness_strategy: str
    # reload-model
    pretrained_model_path: str = None
    total_training_samples: int = 5000

    Early_Stopping: bool = False

    # Gradient-based Filtering
    Ada_Gradient_Settings: List[Any] = ["None", 5]  # labeled Samples path, iters count
    Gaussian_Mixture: bool = False

    # sampling process
    sampling_core: int=4

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


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@auto_config(TaskConfig, __file__)
def main(config: TaskConfig):

    if config.wandb_swith:
        wandb.init(
            # set the wandb project where this run will be logged
            project="{}-{}-{}".format(len(config.xs_selected), config.name, config.ode_model_name),
            group=str(config.name),
            name='seed' + str(config.seed),
            # track hyperparameters and run metadata
            config=config.__dict__
        )

    setup_seed(config.seed)
    # choose cpu or gpu, seed only affect the model parameters
    if config.gpu_idx and config.gpu_idx.isdigit() and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu_idx}')
        torch.cuda.manual_seed_all(config.seed)
    else:
        device = torch.device('cpu')
        torch.manual_seed(config.seed)

    logger.info(f'device: {device}')
    makedir(config.save_dir)

    if config.dataset_type == 'default':
        train_dataset = FileDataSet(
            transform=transform,
            device=device,
            xs_selected=config.xs_selected,
            ys_weight=config.ys_weight,
            # xs_weigth=config.xs_weight,
            norm_xs=config.xs_weight,
            ys_selected=config.ys_selected,
            flag='train',
            *config.train_paths,
            model_name=config.ode_model_name,
        )
    elif config.dataset_type == 'boundary-Only':
        # Gradient-based Filtering Step1
        train_dataset = Only_Boundary_Initialize_Dataset(
            *config.train_paths,
            transform=transform,
            device=device,
            flag='train',
            xs_selected=config.xs_selected,
            ys_weight=config.ys_weight,
            # xs_weigth=config.xs_weight,
            norm_xs=config.xs_weight,
            ys_selected=config.ys_selected,
            sudo_boundary_ratio=config.boundary_sampling_ratio, boundary_K=config.boundary_KNN,
            fitness_strategy=config.fitness_strategy,
            model_name=config.ode_model_name,
        )
    test_dataset = FileDataSet(
        transform=transform,
        device=device,
        xs_selected=config.xs_selected,
        ys_weight=config.ys_weight,
        # xs_weigth=config.xs_weight,
        norm_xs=config.xs_weight,
        ys_selected=config.ys_selected,
        flag='test',
        *config.test_paths,
        model_name=config.ode_model_name,
    )
    # dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # model init
    model = MLPModel(nn_layers=config.nn_layers, nn_norm=config.nn_norm, p=config.dropout,
                     init_strategy=config.init_strategy)
    model.reset_weights()
    model.to(device)

    if config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        model.load_state_dict(torch.load(config.pretrained_model_path), map_location=device)
    else:
        print("NOT FOUND PATH: {}".format(config.pretrained_model_path))
    model.save(os.path.join(config.save_dir, f'model_init.pth'))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=[config.lr])

    if config.train_strategy == 'IS-dag':
        # https://arxiv.org/abs/1803.00942, https://arxiv.org/abs/2302.14772
        from Baseline_Training_code.IS_dag import IS_dag_training
        IS_dag_training(config,
                 model, optimizer, scheduler, train_loader, test_loader,
                 0, 0, 1e9)
    elif config.train_strategy == 'Uniform-fixed':
        from Baseline_Training_code.Training_epoch import training_epochs
        end_epoch = config.last_epoch_n + 0
        iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
            model, optimizer, scheduler, train_loader, test_loader,
            0, 0, 1e9, True,
            0, end_epoch, Early_Stopping=config.Early_Stopping, save_path=None)
    elif config.train_strategy == 'IS':
        # https://arxiv.org/abs/1803.00942, https://arxiv.org/abs/2302.14772
        from Baseline_Training_code.IS import train_IS
        train_IS(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch=0,
          save=True, Early_Stopping=config.Early_Stopping)
    elif config.train_strategy == 'US-S':
        # https://arxiv.org/abs/2307.02719
        # uncertainty sampling based on streaming sampling
        from Baseline_Training_code.RAR_G import RAR_G
        RAR_G(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch=0)
    elif config.train_strategy == 'VeSSAL':
        # https://arxiv.org/abs/2303.02535
        # uncertainty sampling based on streaming sampling
        from Baseline_Training_code.VeSSAL import VeSSAL
        VeSSAL(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch=0)
        
    elif config.train_strategy == 'US-P':
        # https://arxiv.org/abs/2307.02719
        from Baseline_Training_code.UncertaintySampling import UncertaintySampling
        UncertaintySampling(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch=0,
          unlabel_data_path=config.Algorithm_type[0], save_count=config.Algorithm_type[1])
    elif config.train_strategy == 'WRS':
        # https://www.sciencedirect.com/science/article/pii/S002001900500298X
        from Baseline_Training_code.WeightedReservoirSampling import WeightedReservoirSampling
        WeightedReservoirSampling(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch=0,
          unlabel_data_path=config.Algorithm_type[0], save_count=config.Algorithm_type[1])
    elif config.train_strategy == 'HGGS':
        # save unfiltering data
        with open(os.path.join(config.Ada_Gradient_Settings[0]), 'w+') as f:
            length = len(train_dataset.left_inputs)
            lines = []
            for i in range(length):
                fields = train_dataset.left_inputs[i] + train_dataset.left_targets[i]
                line = ",".join([str(_) for _ in fields]) + '\n'
                lines.append(line)
            f.writelines(lines)
        
        N = len(train_dataset) + length
        # compute sampling count each iters for (Gradient-based Filtering Step2)
        target_samples_Ada = int(N * config.uniform_sampling_ratio)
        iters_count_Ada = config.Ada_Gradient_Settings[1]
        sampling_count_Ada = target_samples_Ada // iters_count_Ada

        from Online_Sampling.Ada_Gradient_init import Ada_Gradient_Init
        begin_epoch, iteration, iteration_test, best_model_loss = Ada_Gradient_Init(config,
            model, optimizer, scheduler, train_loader, test_loader,
            iteration=0, iteration_test=0, best_model_loss=1e9,
            begin_epoch=0,
            unlabel_data_path=config.Ada_Gradient_Settings[0], save_count=sampling_count_Ada, iters_count=iters_count_Ada)

        if config.Algorithm_type[0] != 'Uniform-Dynamic':
        # set genetic sampling count
            left_count = config.total_training_samples - len(train_dataset)
            Gene_total_ratio = config.Algorithm_type[1][0] + config.Algorithm_type[1][1]
            CrossOver_ratio = config.Algorithm_type[1][0] / Gene_total_ratio
            Mutation_ratio = config.Algorithm_type[1][1] / Gene_total_ratio

            config.Algorithm_type[1][0] = int(left_count // config.iter_count * CrossOver_ratio)    # CrossOver
            config.Algorithm_type[1][1] = int(left_count // config.iter_count * Mutation_ratio)    # Mutation
            with open(os.path.join(config.save_dir, 'ratio.txt'), 'w+') as f:
                f.write('boundary={}\n'.format(config.boundary_sampling_ratio))
                f.write('uniform={}\n'.format(config.uniform_sampling_ratio))
                f.write('Crossover={}\n'.format(config.Algorithm_type[1][0]))
                f.write('Mutation={}\n'.format(config.Algorithm_type[1][1]))
                f.write('Online-Algorithm={}\n'.format(config.Algorithm_type))

        # Genetic-Sampling phase
        from Online_Sampling.Genetic_Sampling import Genetic_Training
        model = MLPModel(nn_layers=config.nn_layers, nn_norm=config.nn_norm, p=config.dropout,
                 init_strategy=config.init_strategy)
        model.reset_weights()
        model.to(device)
        optimizer = optim.Adam(model.parameters(),
                        config.base_lr,
                        weight_decay=config.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=[config.lr])
        Genetic_Training(config,
                 model, optimizer, scheduler, train_loader, test_loader,
                 iteration=0, iteration_test=0, best_model_loss=1e9,
                 begin_epoch=0)

if __name__ == '__main__':
    main()
    exit(0)
