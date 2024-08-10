"""
@author:
@email:
@date: 2023-8-19
@description:
"""
import sys
import wandb

sys.path.extend(['../../../../src'])    # noqa
from yu.tasks.sample.combine_sample import run_
from yu.tools.pre_produce_data import produce_data

from typing import List
from pydantic import validator
from yu.const.grid import SearchType
from yu.core.config import BaseConfig, auto_config
from yu.tasks.pde_models import Truth
import os
import time
import numpy as np

from yu.const.normal import PDEType, RandomType, DisplayType
from yu.core.logger import logger
from yu.tools.grid import Grid
from yu.tools.misc import sample_lhs, makedir
from yu.tools.plot import save_plot, show_plot

class GeneratorConfig(BaseConfig):
    total_good: int
    total_bad: int
    n_good: int = 0
    n_bad: int = 0

    keep_single: bool
    display: DisplayType = DisplayType.NONE

    random_type: RandomType
    threshold: List[float]
    params_selected: List[int]

    # RandomType.LHS
    dimension: int
    log_lb: float
    log_ub: float
    use_log: bool
    per_split_n: List[int]  # 
    max_split_times: int = 0
    init_split_n: int
    cube_n: List[float]
    search_type: SearchType

    # RandomType.RANDOM
    r_rate: List[List[float]] = None

    good_dir: str = 'good'
    bad_dir: str = 'bad'
    good_f: str = 'good.csv'
    bad_f: str = 'bad.csv'

    db_path: str = 'grid'
    tablename: str = 'generate_data'

    # Multi-Process
    data_path: str
    core: int
    reload: bool = False
    reload_path: str = None

    @validator('good_dir', pre=True)
    def set_good_dir(cls, v, values):
        root_dir = values.get('save_dir')
        good_dir = os.path.join(root_dir, v)
        makedir(good_dir)
        return good_dir

    @validator('bad_dir', pre=True)
    def set_bad_dir(cls, v, values):
        root_dir = values.get('save_dir')
        bad_dir = os.path.join(root_dir, v)
        makedir(bad_dir)
        return bad_dir

    @validator('good_f', pre=True)
    def set_good_f(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('bad_f', pre=True)
    def set_bad_f(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('db_path', pre=True)
    def set_db_path(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('max_split_times', pre=True)
    def set_max_split_times(cls, v, values):
        return len(values.get('per_split_n'))

def run_process_generate_data(data_path, core, config, truth, _y0, _T, _T_unit):
    '''
    process Multi-Procession generate_data.py
    '''
    run_(data_path, core, config, truth, _y0, _T, _T_unit)

@auto_config(GeneratorConfig, __file__)
def main(config: GeneratorConfig, truth: Truth, _y0, _T, _T_unit):
    """"""
    '''
    data save_path: data_path='../../../../{}param/sub{}/'
    '''
    if not config.reload:
        produce_data(config)

    run_process_generate_data(config.data_path.format(len(config.params_selected), config.core)
                              , config.core, config, truth, _y0, _T, _T_unit)

# HSECC
from yu.tasks.model_HSECC import Parameters as HSECC_Params
from yu.tasks.model_HSECC import Equation as HSECC_Equs
# PS2_01
from yu.tasks.model_PS2_01 import Parameters as PS2_01_Params
from yu.tasks.model_PS2_01 import Equation as PS2_01_Equs
# brusselator
from yu.tasks.model_brusselator import Parameters as brusselator_Params
from yu.tasks.model_brusselator import Equation as brusselator_Equs
# MPF_2_Var
from yu.tasks.model_MPF_2_Var import Parameters as MPF_2_Var_Params
from yu.tasks.model_MPF_2_Var import Equation as MPF_2_Var_Equs


if __name__ == '__main__':
    # model_name = 'PS2_01'
    model_name = 'brusselator'
    # model_name = 'MPF_2_Var'

    if model_name == 'HSECC':
        Parameters = HSECC_Params
        Equation = HSECC_Equs

        model_name = model_name + ' Cell Cycle'
        equation_indices = [1, 3, 4]
        curve_names = ['X', 'Y', 'Z']
        curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        _y0 = [30, 320, 100, 100, 200]
        _T = 1000
        _T_unit = 1
    elif model_name == 'PS2_01':
        Parameters = PS2_01_Params
        Equation = PS2_01_Equs

        model_name = model_name + ' Cycle'
        equation_indices = [0, 1]
        curve_names = ["x", "y"]
        curve_colors = ['#1f77b4', '#ff7f0e']
        _y0 = [1, 4]
        _T = 5000
        _T_unit = 1
    elif model_name == 'brusselator':
        Parameters = brusselator_Params
        Equation = brusselator_Equs

        model_name = model_name + ' Cycle'
        equation_indices = [0, 1]
        curve_names = ["x", "y"]
        curve_colors = ['#1f77b4', '#ff7f0e']
        
        _y0 = [10, 10]
        _T = 500
        _T_unit = 1e-2

    elif model_name == 'MPF_2_Var':
        Parameters = MPF_2_Var_Params
        Equation = MPF_2_Var_Equs

        model_name = model_name + ' Cycle'
        equation_indices = [0, 1]
        curve_names = ["x", "y"]
        curve_colors = ['#1f77b4', '#ff7f0e']
        
        _y0 = [0.03656777, 0.36614645]
        _T = 1000
        _T_unit = 1e-2


    truth = Truth()
    truth.config(
        model_name=model_name,
        curve_names=curve_names,
        curve_colors=curve_colors,
        equation_idx=equation_indices,
        threshold=0.40,
        equs=Equation,
        _Parameters=Parameters,
    )
    main(truth=truth, _y0=_y0, _T=_T, _T_unit=_T_unit)
    exit(0)
