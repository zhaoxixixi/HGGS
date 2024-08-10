import sys
sys.path.extend(['../../../../../src'])
import torch
import os
from yu.tasks.OnlineSamplingTools.Config import TaskConfig
from yu.nn.EarlyStopping import EarlyStopping
from yu.tasks.OnlineSamplingTools.Baseline_Training_code.Training_epoch import training_epochs

import wandb
import numpy as np

def get_weight(model, train_loader):
    # get gradient-weight
    model.eval()

    grad_list = []
    indices_list = []
    for i, (inputs, targets, indices) in enumerate(train_loader):
        outputs = model(inputs)
        with torch.no_grad():
            grad_norm = (torch.norm(2 * (outputs - targets), dim=-1).detach().cpu().numpy())  # grad_norm

        grad_list.extend(grad_norm)
        indices_list.extend(indices)

    weights = np.zeros(shape=(len(indices_list),))
    for _idx, _grad_norm in zip(indices_list, grad_list):
        weights[_idx] = _grad_norm

    return weights


def Genetic_Training(config: TaskConfig,
                 model, optimizer, scheduler, train_loader, test_loader,
                 iteration=0, iteration_test=0, best_model_loss=1e9,
                 begin_epoch: int=None):
    Gaussian_Mixture = config.Gaussian_Mixture
    global_early_stopping = EarlyStopping(save_path=config.save_dir, patience=5, delta=-1e-6)


    # warm up training
    end_epoch = config.last_epoch_n + begin_epoch
    iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
        model, optimizer, scheduler, train_loader, test_loader,
        iteration, iteration_test, best_model_loss, True,
        begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping, save_path=None)
    global_early_stopping(last_model_loss, model)

    from yu.tasks.OnlineSamplingTools.Sampling_Algorithm.genetic_T3 import Gene_T3_Thread

    # init generate Algorithm
    if config.Algorithm_type[0] == 'Gene_T3_Thread':
        gene = Gene_T3_Thread('Gene_T3_Thread', config.Algorithm_type,
            config.xs_param, config.xs_weight, config.param_selected,
                              max_threads_at_once=config.sampling_core, ode_model_name=config.ode_model_name)
    elif config.Algorithm_type[0] == "Uniform-Dynamic":
        from init_algorithm import Uniform_Algorithm
        # ['Uniform', max_process, sampling_count, save_number]
        gene = Uniform_Algorithm(config.xs_weight, config.xs_param, config.param_selected, 
                                    max_process=config.Algorithm_type[1], ode_model_name=config.ode_model_name)

    else:
        assert False, 'Error Algorithm_type'

    # Online Sampling begin
    cur_xs_selected = []
    cur_ys_selected = []
    for i in range(len(config.ys_selected)):
        cur_ys_selected.append(4 * (i + 1))

    for i in range(len(config.xs_selected)):
        cur_xs_selected.append(config.param_selected[i] + cur_ys_selected[-1] + 1)

    end_epoch = begin_epoch + config.epoch_n
    for _ in range(config.iter_count):
        
        # stratified residual
        inputs = np.array(train_loader.dataset._inputs.copy())
        weights = get_weight(model, train_loader)

        if config.Algorithm_type[0] == 'Uniform-Dynamic':
            new_inputs, new_targets = gene.generate_samples(config.Algorithm_type[2], 
                cur_xs_selected, cur_ys_selected, config.ys_weight, None)
        else:
            new_inputs, new_targets = gene.generate_samples(inputs, weights,
                cur_xs_selected, cur_ys_selected, config.ys_weight, Gaussian_Mixture=Gaussian_Mixture)

        train_loader.dataset.residule_new_data(new_inputs, new_targets)

        cur_save_path = os.path.join(config.save_dir, str(begin_epoch))
        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)

        iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
            model, optimizer, scheduler, train_loader, test_loader,
            iteration, iteration_test, best_model_loss, log=True,
            begin_epoch=begin_epoch, end_epoch=end_epoch, Early_Stopping=config.Early_Stopping)
        end_epoch = begin_epoch + config.epoch_n

        # save dataset this training
        with open(os.path.join(cur_save_path, 'current_data.csv'), 'w+') as f:
            lines = []
            for i in range(len(train_loader.dataset._inputs)):
                line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)
        if global_early_stopping:
            global_early_stopping(last_model_loss, model)
            if global_early_stopping.early_stop:
                with open(os.path.join(config.save_dir, 'iters.txt'), 'w+') as f:
                    f.write("Early Stopping in Ada-iters {}\n".format(_))
                break

    # Online Sampling end
    with open(os.path.join(config.save_dir, 'data.csv'), 'w+') as f:
        lines = []
        for i in range(len(train_loader.dataset._inputs)):
            line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
            lines.append(line)

        f.writelines(lines)
