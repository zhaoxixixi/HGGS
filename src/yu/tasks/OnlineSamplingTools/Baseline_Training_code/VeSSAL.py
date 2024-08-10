import sys
from .Training_epoch import training_epochs
sys.path.extend(['../../../../../'])
import torch
import os
import numpy as np

def inf_replace(mat):
    mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
    return mat

def predict_model(model, inputs, targets, device, dtype, budget,
                  cov_inv_scaling: float=100) -> tuple[list[any], list[any]]:
    # code borrowed from https://github.com/asaran/VeSSAL/blob/main/query_strategies/vessal.py
    inputs = torch.tensor(inputs, device=device, dtype=dtype)
    targets = torch.tensor(targets, device=device, dtype=dtype)

    grad_embedding = model.get_last_layer_grad(inputs, targets)
    dim = grad_embedding.shape[-1]
    rank = 1

    covariance = torch.zeros(dim,dim, device=device, dtype=dtype)
    covariance_inv = cov_inv_scaling * torch.eye(dim, device=device, dtype=dtype)
    samps = torch.tensor(grad_embedding, device=device, dtype=dtype)

    save_indices = []
    for i, u in enumerate(samps):
        u = u.view(-1, 1)
        
        # get determinantal contribution (matrix determinant lemma)
        norm = torch.abs(u.t() @ covariance_inv @ u)

        ideal_rate = (budget - len(save_indices))/(len(samps) - (i))
        # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
        covariance = (i/(i+1))*covariance + (1/(i+1))*(u @ u.t())

        zeta = (ideal_rate/(torch.trace(covariance @ covariance_inv))).item()

        pu = np.abs(zeta) * norm

        if np.random.rand() < pu.item():
            save_indices.append(i)
            if len(save_indices) >= budget:
                break
            
            # woodbury update to covariance_inv
            inner_inv = torch.inverse(torch.eye(rank, device=device, dtype=dtype) + u.t() @ covariance_inv @ u)
            inner_inv = inf_replace(inner_inv)
            covariance_inv = covariance_inv - covariance_inv @ u @ inner_inv @ u.t() @ covariance_inv

    return inputs[save_indices].detach().cpu().tolist(), targets[save_indices].detach().cpu().tolist()


def VeSSAL(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch: int=1):
    from yu.nn.EarlyStopping import EarlyStopping
    if config.Early_Stopping:
        early_stopping = EarlyStopping(save_path=config.save_dir, patience=5)
    else:
        early_stopping = None

    if config.Algorithm_type[0] != 'Uniform':
        assert False, 'Sampling METHOD is not Uniform!\n'
        exit(0)

    # ----------------------------------train begin----------------------------------
    end_epoch = config.last_epoch_n + begin_epoch
    iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration, iteration_test, best_model_loss, True,
          begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping)
    end_epoch = begin_epoch + config.epoch_n
    
    early_stopping(last_model_loss, model)

    from init_algorithm import Uniform_Algorithm
    # ['Uniform', max_process, sampling_count, save_number]
    gene = Uniform_Algorithm(config.xs_weight, config.xs_param, config.param_selected, 
                                 max_process=config.Algorithm_type[1], ode_model_name=config.ode_model_name)
    
    cur_xs_selected = []
    cur_ys_selected = []
    for i in range(len(config.ys_selected)):
        cur_ys_selected.append(4 * (i + 1))

    for i in range(len(config.xs_selected)):
        cur_xs_selected.append(config.param_selected[i] + cur_ys_selected[-1] + 1)

    train_dataset = train_loader.dataset
    for _ in range(config.iter_count):
        # sampling by https://arxiv.org/abs/2303.02535
        '''
        1. uniform sampling
        2. calculate pt for each samples
        3. ac the sample by pt
        4. if total sample size not reach budget, then continue sampling&training, otherwise go 5
        5. End continual learning
        '''
        new_inputs, new_targets = gene.generate_samples(config.Algorithm_type[2], 
                cur_xs_selected, cur_ys_selected, config.ys_weight, None)
        
        budget = min(config.total_training_samples - len(train_dataset), len(new_inputs))
        save_inputs, save_targets = predict_model(model, new_inputs, new_targets,
                                                  train_dataset.device, torch.float64, budget)

        # add new data
        train_dataset.residule_new_data(save_inputs, save_targets)

        cur_save_dir = os.path.join(config.save_dir, str(begin_epoch))
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        
        with open(os.path.join(cur_save_dir, 'residual_data.csv'), 'w+') as f:
            lines = []
            for i in range(len(save_inputs)):
                line = ",".join([str(_) for _ in (save_inputs[i] + save_targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)

        iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
            model, optimizer, scheduler, train_loader, test_loader,
            iteration, iteration_test, best_model_loss, True,
            begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping)
        end_epoch = begin_epoch + config.epoch_n

        
        if early_stopping:
            early_stopping(last_model_loss, model)
            if early_stopping.early_stop:
                with open(os.path.join(cur_save_dir, 'EarlyStop.txt'), 'w+') as f:
                    f.write("Early Stopping in epoch {}\n".format(_))
                break

    with open(os.path.join(config.save_dir, 'data.csv'), 'w+') as f:
        lines = []
        for i in range(len(train_loader.dataset._inputs)):
            line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
            lines.append(line)

        f.writelines(lines)
