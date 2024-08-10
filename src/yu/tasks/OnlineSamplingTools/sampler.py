"""
@author:
@email:
@date:
@description: Data to GroundTruth
"""
import numpy as np
import os
import sys
sys.path.extend(['../../../../src'])
from yu.tasks.sample.generate_data import save_img
from yu.tasks.pde_models import Truth

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

from yu.const.normal import PDEType, DisplayType
from yu.core.logger import logger
from yu.tools.misc import makedir


from yu.tasks.pde_models import transform
def get_truth_line(data: list, model_name: str,
                   save_fig: bool=False, save_path: str=None,
                   y0: list=None):
    truth = Truth()


    if model_name == 'HSECC':
        Parameters = HSECC_Params
        Equation = HSECC_Equs

        model_name = model_name + ' Cell Cycle'
        equation_indices = [1, 3, 4]
        curve_names = ['X', 'Y', 'Z']
        curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        _y0 = [30, 320, 100, 100, 200]
        _T = 1000
        # _T = 100
        _T_unit = 1
    elif model_name == 'PS2_01':
        Parameters = PS2_01_Params
        Equation = PS2_01_Equs

        model_name = model_name + ' Cycle'
        equation_indices = [0, 1]
        curve_names = ["x", "y"]
        curve_colors = ['#1f77b4', '#ff7f0e']
        if y0 is not None:
            _y0 = y0
        else:
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
        
        if y0 is not None:
            _y0 = y0
        else:
            _y0 = [10, 10]
        _T = 500
        # _T = 50
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
    

    # print(_y0)
    truth.config(
        model_name=model_name,
        curve_names=curve_names,
        curve_colors=curve_colors,
        equation_idx=equation_indices,
        threshold=0.40,
        equs=Equation,
        _Parameters=Parameters,
    )

    params = Parameters()
    params.assign(data)

    truth.calc(
        y0=np.asarray(_y0),
        T=_T,
        T_unit=_T_unit,
        params=params,
    ).find_period()

    # get GroundTruth

    # print(truth.success)
    if not truth.success and truth.pde_type == PDEType.NAN:
        logger.error('Calculation failure because of NAN.')
        return None
    
    # only success
    if truth.success and save_fig:
        # save img
        save_img(
            truth=truth,
            model_name=model_name,
            png_name=os.path.join('sampler_test', model_name + '.png') if save_path is None else os.path.join(save_path, model_name + '.png'),
            keep_single=True,
            display=DisplayType.SAVE,
        )

    if truth.pde_type == PDEType.OSCILLATION:
        model_name = 'g_1'
    else:
        model_name = 'b_1'
    # save data
    csv_line = params.build_csv(
        truth=truth,
        params=params,
        name=model_name,
    )

    return csv_line

if __name__ == '__main__':
    # HSECC
    model_name = 'HSECC'
    x = [1.53,0.04,5.4,0.15,1.35,0.1,0.00741]
    xs_selected = [15, 16]
    ys_selected = [4, 8, 12]
    ys_weights = [1, 1, 1]

    print(get_truth_line(x, model_name=model_name, y0 = None))
    xs, ys = transform(get_truth_line(x, model_name=model_name, save_fig=True, y0 = None), 
                    xs_selected, ys_selected, ys_weights, None, model_name=model_name)
    print(xs, ys)
    exit(0)
