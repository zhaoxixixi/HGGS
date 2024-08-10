
import os
import numpy as np
import sys
sys.path.extend(['../../../../src'])  # noqa
# from yu.nn.dataset import FileDataSet
from yu.tasks.OnlineSamplingTools.dataloader import FileDataSet

from yu.tasks.pde_models import transform
from yu.nn.model import MLPModel, get_model
from yu.tools.plot import draw_scatter
from yu.const.normal import FigureType, DisplayType

import torch
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="US-P-5k")  # train data
parser.add_argument('--model_mode', type=str)  # model path
parser.add_argument('--ode_model_name', type=str, default="HSECC")  # model path
parser.add_argument('--xs_param', type=str, default="[1.53, 0.04, 1.35, 0.02, 1.35, 0.1, 0.00741]")  # model path
parser.add_argument('--xs_param_selected', type=str, default="[0, 1, 2, 3, 4, 5]")  # model path
parser.add_argument('--xs_selected', type=str, default="[0, 1, 2, 3, 4, 5]")  # model path
parser.add_argument('--ys_selected', type=str, default="[6, 7, 8]")  # model path

args = parser.parse_args()

# seeds = [53,25,81,99,50]
seeds = [53]

nn_model = 'MLP'
device = 'cpu'
model_selected = 'best_network'

model_name = args.model_name
model_mode = args.model_mode
ode_model_name = args.ode_model_name
xs_param = eval(args.xs_param)
xs_param_selected = eval(args.xs_param_selected)
xs_selected = eval(args.xs_selected)
ys_selected = eval(args.ys_selected)

# HSECC
# ode_model_name = 'HSECC'
# xs_param=[1.53, 0.04, 1.35, 0.02, 1.35, 0.1, 0.00741]
# xs_param_selected = [0, 1, 2, 3, 4, 5]
# xs_selected = [0, 1, 2, 3, 4, 5]
# ys_selected=[6, 7, 8]

# brusselator
# ode_model_name = 'brusselator'
# xs_param=[1, 3]
# xs_param_selected = [0, 1]
# xs_selected = [0, 1]
# ys_selected=[2, 3]

# MPF_2_Var 6Param
# ode_model_name = 'MPF_2_Var'
# xs_param = [0.01, 0.01, 10, 0.04, 100, 1.5, 0.1, 1]
# xs_param_selected = [0, 1, 3, 5, 6, 7]
# xs_selected = [0, 1, 2, 3, 4, 5]
# ys_selected=[6, 7]

# PS2_01
# ode_model_name = 'PS2_01'
# xs_param = [2.8, 0.1, 0.1, 1, 5, 1]
# xs_param_selected = [0, 1, 2, 3, 4, 5]
# xs_selected = [0, 1, 2, 3, 4, 5]
# ys_selected=[6, 7]

if ode_model_name == 'MPF_2_Var':
    x_lim = [0, 0.3]
elif ode_model_name == 'PS2_01':
    x_lim = [0, 0.1]
elif ode_model_name == 'HSECC':
    x_lim = [0, 0.2]
else:
    x_lim = [0, 1.0]

xs_lb_ub=[0,10]
if ode_model_name == 'brusselator':
    xs_lb_ub = [0, 5]

param = len(xs_selected)
xs_weight = [[xs_lb_ub[0] * xs_param[i], xs_lb_ub[1] * xs_param[i]] for i in range(len(xs_param))]
rev_output_norm = [xs_param[i] * (xs_lb_ub[1] - xs_lb_ub[0]) for i in range(len(xs_param))]
norm_Min = [xs_param[i] * (xs_lb_ub[0]) for i in range(len(xs_param))]

xs_weight = np.array(xs_weight)[xs_param_selected].tolist()
rev_output_norm = np.array(rev_output_norm)[xs_param_selected].tolist()
norm_Min = np.array(norm_Min)[xs_param_selected].tolist()

if ode_model_name == 'HSECC' or ode_model_name == 'brusselator':
    nn_layers = [len(xs_selected), 128, 256, 128, len(ys_selected)]
elif ode_model_name == 'MPF_2_Var':
    nn_layers = [len(xs_selected), 128, 128, 128, 128, len(ys_selected)]
elif ode_model_name == 'PS2_01':
    nn_layers = [len(xs_selected), 256, 256, 256, 256, len(ys_selected)]

data_name = [
    'test_all',
    'test_NonOsci_all',
    'test_Osci_all',
    'test_boundary_all',
]
BN = ["BatchNorm1d"] * (len(nn_layers) - 2)

if ode_model_name == 'brusselator':
    model = get_model(nn_model, nn_layers, BN, 0.0, 
                    'xiver')
else:
    model = get_model(nn_model, nn_layers, BN, 0.2, 
                    'xiver')
model = model.to(device)

def model_test(test_name):
    cur_ode_model_name = ode_model_name
    test_path = r'../../../../data/6param/{}/test/{}.csv'.format(ode_model_name, test_name)
    if ode_model_name == 'brusselator':
        test_path = r'../../../../data/2param/{}/test/{}.csv'.format(ode_model_name, test_name)

    test_dataset = FileDataSet(
            transform=transform,
            device=device,
            xs_selected=xs_selected,
            ys_selected=ys_selected,
            norm_xs=xs_weight,
            flag='test',
            *[test_path],
            model_name=cur_ode_model_name,
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )
    from yu.tools.misc import get_inputs_targets_outputs
    Matrix = [0., 0., 0.]  # MAE, MSE, RMSE

    inputs, targets, outputs = get_inputs_targets_outputs(model, test_loader)
    targets = np.array(targets)
    outputs = np.array(outputs)
    residual = np.mean(np.abs(targets - outputs), axis=-1)

    cur_MAE = np.mean(residual)
    cur_MSE = np.mean(residual ** 2)
    cur_RMSE = np.sqrt(cur_MSE)
    Matrix[0] = cur_MAE
    Matrix[1] = cur_MSE
    Matrix[2] = cur_RMSE

    print('Max - Min =', rev_output_norm)
    print('Min =', norm_Min)
    print('max_residual {}: {}'.format(test_name, np.max(residual)))

    return Matrix

def get_mean(result):
    return np.mean(result)
def get_cv(result):
    return np.std(result) / np.mean(result)
def get_ratio(result):
    mean_data = get_mean(result)
    max_data = np.max(result)
    min_data = np.min(result)
    
    max_ratio = abs(max_data - mean_data) / mean_data
    min_ratio = abs(min_data - mean_data) / mean_data
    return max(max_ratio, min_ratio)

if __name__ == '__main__':
    RMSE_Result = []

    for test_name in data_name:
        seeds_result = [[], [], []]  # MAE, MSE, RMSE
        save_png_path = './result-seeds/{}/{}/{}'.format(test_name, model_mode, model_name)
        for seed in seeds:
            if ode_model_name == 'brusselator':
                model_path = r'../../../../output/mlp/reg_2/2param/{}/{}/{seed}/{}.pth'.format(model_mode, model_name, model_selected, seed=seed)
            else:
                model_path = r'../../../../output/mlp/reg_2/6param/{}/{}/{seed}/{}.pth'.format(model_mode, model_name, model_selected, seed=seed)
            model.load_state_dict(torch.load(model_path))
            Matrix = model_test(test_name)

            for j in range(len(Matrix)):
                seeds_result[j].append(Matrix[j])

        Matrix_name = ['MAE', 'MSE', 'RMSE']  # MAE, MSE, RMSE
        csv_name = 'Categories/Seeds-Mean-CV'

        if not os.path.exists(save_png_path):
            os.makedirs(save_png_path)
        with open(os.path.join(save_png_path, 'Loss.csv'), 'w+') as f:
            f.write(csv_name + ',' + ','.join([str(_) for _ in seeds]) + ',Mean,CV' + '\n')
            for j in range(len(Matrix_name)):
                f.write(Matrix_name[j] + ',' + ','.join([str(_) for _ in seeds_result[j]]) + ',' + 
                str(get_mean(seeds_result[j])) + ',' + str(get_cv(seeds_result[j])) + '\n')

        RMSE_Result.append([])
        RMSE_Result[-1].append(test_name)
        RMSE_Result[-1].append("{:.4f}".format(get_mean(seeds_result[-1])))

        RMSE_Result[-1].append("{:.4f}".format(np.std(seeds_result[-1], ddof=1)))
    
    save_combine_result_path = r'./result.csv'
    RMSE_Result = np.array(RMSE_Result)
    print("testing model:", model_name)
    print("RMSE results:")
    print(RMSE_Result)
    with open(save_combine_result_path, 'w+') as f:
        f.writelines(",".join(["Model/Type"] + [str(name) for name in RMSE_Result[:, 0]]) + '\n')
        f.writelines(",".join([model_name] + [str(RMSE_RATIO[0])[:6] + "±" + str(RMSE_RATIO[1])[:6] for RMSE_RATIO in RMSE_Result[:, 1:3]]) + '\n')
