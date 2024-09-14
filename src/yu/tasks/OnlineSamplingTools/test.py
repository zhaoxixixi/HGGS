import os
import numpy as np
import sys
sys.path.extend(['../../../../src'])  # noqa
from yu.tools.misc import get_sampling_name, get_mean, get_cv, get_ratio
from yu.tasks.OnlineSamplingTools.dataloader import FileDataSet
from yu.tasks.BioSysConfig.BioSysConfig import SysConfig
from yu.tools.misc import get_inputs_targets_outputs
from yu.tasks.pde_models import transform
from yu.nn.model import MLPModel, get_model
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_set', type=str, default="US-P-5k")  # train data
parser.add_argument('--model_path', type=str)  # model path
parser.add_argument('--seeds', type=str, default="[53]")  # seeds
parser.add_argument('--ode_model_name', type=str, default="HSECC")  # biological system name

args = parser.parse_args()

sampling_name = get_sampling_name(args.train_set)
model_selected = 'best_network'

model_name = args.train_set
model_mode = args.model_path
ode_model_name = args.ode_model_name
seeds = eval(args.seeds)
system_config = SysConfig(sampling_name, args.ode_model_name)

xs_param = system_config.xs_param
xs_selected = system_config.xs
ys_selected = system_config.ys
xs_lb_ub = system_config.xs_lb_ub
# model settings
nn_layers = system_config.nn_layers
BN = system_config.nn_norm

xs_weight = [[xs_lb_ub[0] * xs_param[i], xs_lb_ub[1] * xs_param[i]] for i in range(len(xs_param))]
xs_weight = np.array(xs_weight)[system_config.param_selected].tolist()

data_name = [
    'test_all',
    'test_NonOsci_all',
    'test_Osci_all',
    'test_boundary_all',
]

model = get_model('MLP', nn_layers, BN, system_config.dropout, 'xiver')
model = model.to('cpu')

def model_test(test_name):
    cur_ode_model_name = ode_model_name
    test_path = r'../../../../data/{}param/{}/test/{}.csv'.format(len(xs_selected), ode_model_name, test_name)

    test_dataset = FileDataSet(
            transform=transform,
            device='cpu',
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
    Matrix = [0., 0., 0.]  # MAE, MSE, RMSE
    inputs, targets, outputs = get_inputs_targets_outputs(model, test_loader)
    targets = np.array(targets)
    outputs = np.array(outputs)
    residual = np.mean(np.abs(targets - outputs), axis=-1)
    Matrix[0] = np.mean(residual)
    Matrix[1] = np.mean(residual ** 2)
    Matrix[2] = np.sqrt(Matrix[1])

    print('max_residual {}: {}'.format(test_name, np.max(residual)))
    return Matrix

if __name__ == '__main__':
    RMSE_Result = []

    for test_name in data_name:
        seeds_result = [[], [], []]  # MAE, MSE, RMSE
        save_png_path = './result-seeds/{}/{}/{}'.format(test_name, model_mode, model_name)
        for seed in seeds:
            model_path = r'../../../../output/mlp/reg_2/{}param/{}/{}/{seed}/{}.pth'.format(len(xs_selected), model_mode, model_name, model_selected, seed=seed)
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

        if len(seeds_result[-1]) > 1:
            RMSE_Result[-1].append("{:.4f}".format(np.std(seeds_result[-1], ddof=1)))
        else:
            RMSE_Result[-1].append("{:.4f}".format(0.0))
    
    save_combine_result_path = r'./result.csv'
    RMSE_Result = np.array(RMSE_Result)
    print("testing model:", model_name)
    print("RMSE results:")
    print(RMSE_Result)
    with open(save_combine_result_path, 'w+') as f:
        f.writelines(",".join(["Model/Type"] + [str(name) for name in RMSE_Result[:, 0]]) + '\n')
        f.writelines(",".join([model_name] + [str(RMSE_RATIO[0])[:6] + "±" + str(RMSE_RATIO[1])[:6] for RMSE_RATIO in RMSE_Result[:, 1:3]]) + '\n')
