# -*- coding: utf-8 -*-
# @Time    : 2024/9/14 17:31
# @Author  :
# @FileName: BioSysConfig.py.py
# @Software: PyCharm
# @Blog    :
# descript : training / testing configuration

class SysConfig():
    def __init__(self, sampling_name: str, sys_name: str):
        assert sys_name in ['brusselator', 'HSECC', 'MPF_2_Var', 'PS2_01'], \
            'Error! Only define `[brusselator, HSECC, MPF_2_Var, PS2_01]`'
        assert sampling_name in ['HGGS', 'IS', 'IS-dag', 'US-P', 'US-S', 'VeSSAL', 'WRS']

        train_data_mode = "GG-Sampling" if sampling_name == 'HGGS' else "Baseline"
        self.sys_name = sys_name
        if sys_name == 'brusselator':
            self.xs = [0,1]
            self.ys = [2,3]
            self.data_path = f"/brusselator/train/{train_data_mode}/"
            self.ode_model_name = 'brusselator'
            self.test_path = "../../../../data/2param/brusselator/val/val_5k.csv"
            self.xs_param = [1, 3]
            self.param_selected = [0,1]
            self.xs_lb_ub = [0, 5]
            self.nn_layers = [128, 256, 128]

            self.nn_norm = "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]"
            self.dropout = 0.0

            self.x_lim = [0, 1.0]

        elif sys_name == 'HSECC':
            self.xs = [0, 1, 2, 3, 4, 5]
            self.ys = [6, 7, 8]
            self.data_path = f"/HSECC/train/{train_data_mode}/"
            self.ode_model_name = 'HSECC'
            self.test_path = "../../../../data/6param/HSECC/val/val_5k.csv"
            self.xs_param = [1.53,0.04,1.35,0.02,1.35,0.1,0.00741]
            self.param_selected = [0,1,2,3,4,5]
            self.xs_lb_ub = [0, 10]
            self.nn_layers = [128, 256, 128]

            self.nn_norm = "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]"
            self.dropout = 0.15

            self.x_lim = [0, 0.2]

        elif sys_name == 'MPF_2_Var':
            self.xs = [0, 1, 2, 3, 4, 5]
            self.ys = [6, 7]
            self.data_path = f"/MPF_2_Var/train/{train_data_mode}/"
            self.ode_model_name = 'MPF_2_Var'
            self.test_path = "../../../../data/6param/MPF_2_Var/val/val_5k.csv"
            self.xs_param = [0.01, 0.01, 10, 0.04, 100, 1.5, 0.1, 1]
            self.param_selected = [0,1,3,5,6,7]
            self.xs_lb_ub = [0, 10]
            self.nn_layers = [128, 128, 128, 128]

            self.nn_norm = "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]"
            self.dropout = 0.12

            self.x_lim = [0, 0.3]

        elif sys_name == 'PS2_01':
            self.xs = [0, 1, 2, 3, 4, 5]
            self.ys = [6, 7]
            self.data_path = f"/PS2_01/train/{train_data_mode}/"
            self.ode_model_name = 'PS2_01'
            self.test_path = "../../../../data/6param/PS2_01/val/val_5k.csv"
            self.xs_param = [2.8, 0.1, 0.1, 1, 5, 1]
            self.param_selected = [0,1,2,3,4,5]
            self.xs_lb_ub = [0, 10]
            self.nn_layers = [256, 256, 256, 256]

            self.nn_norm = "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]"
            self.dropout = 0.1

            self.x_lim = [0, 0.1]

        self.lr_alpha = 0.2
        self.Gaussian_Mixture = True
        self.Ada_Gradient_Settings = "[\"None\", 5]"

        # sampling method
        if sampling_name == 'HGGS':
            self.iter_count = 20
            self.Algorithm_type = "[\"Gene_T3_Thread\", [6.0, 4.0], [\"A\", \"D\"]]"

            self.dataset_type = "boundary-Only"
            self.uniform_sampling_ratio = 0.30
            self.boundary_sampling_ratio = 0.20
            self.boundary_KNN = 5

            # Importance Sampling training Settings
            self.finetune_epoch = 0
            self.finetune_count = 0
            self.tau = -1

            self.train_strategy = 'HGGS'
        elif sampling_name == 'IS':
            self.iter_count = 20
            self.Algorithm_type = "[]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 3000
            self.finetune_count = 20
            self.tau = -1
            self.train_strategy = 'IS'
        elif sampling_name == 'IS-dag':
            self.iter_count = 20
            self.Algorithm_type = "[]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 3000
            self.finetune_count = 20
            self.tau = -1
            self.train_strategy = 'IS-dag'
        elif sampling_name == 'US-P':
            self.iter_count = 20
            self.Algorithm_type = f"[\"../../../../data/{len(self.xs)}param/{self.sys_name}/train/Baseline/US-P/pool-1w.csv\", 250]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 0
            self.finetune_count = 0
            self.tau = -1
            self.train_strategy = 'US-P'

        elif sampling_name == 'US-S':
            self.iter_count = 20
            self.Algorithm_type = "[\"Uniform\", 32, 500, 250]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 0
            self.finetune_count = 0
            self.tau = -1
            self.train_strategy = 'US-S'

        elif sampling_name == 'VeSSAL':
            self.iter_count = 20
            self.Algorithm_type = "[\"Uniform\", 32, 500, 250]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 0
            self.finetune_count = 0
            self.tau = -1
            self.train_strategy = 'VeSSAL'

        elif sampling_name == 'WRS':
            self.iter_count = 20
            self.Algorithm_type = f"[\"../../../../data/{len(self.xs)}param/{self.sys_name}/train/Baseline/WRS/pool-5k.csv\", 250]"

            self.dataset_type = "default"
            self.uniform_sampling_ratio = 0.30  # useless
            self.boundary_sampling_ratio = 0.20  # useless
            self.boundary_KNN = 5  # useless

            # Importance Sampling training Settings
            self.finetune_epoch = 0
            self.finetune_count = 0
            self.tau = -1
            self.train_strategy = 'WRS'
