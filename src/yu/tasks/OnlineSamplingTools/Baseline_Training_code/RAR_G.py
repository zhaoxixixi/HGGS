import sys
from .Training_epoch import training_epochs
sys.path.extend(['../../../../../'])
import torch
import os
import numpy as np

def predict_model(model, inputs, targets, device, dtype):
    model.eval()
    inputs = torch.tensor(inputs, device=device, dtype=dtype)
    targets = np.array(targets)
    
    outputs = model(inputs).cpu().detach().numpy()

    return abs(outputs - targets)[:, 0]


def RAR_G(config,
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

    # residual timing
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
        # sampling by https://github.com/lu-group/pinn-sampling/blob/main/src/allen_cahn/RAR_G.py
        '''
        X = geomtime.random_points(100000)
        Y = np.abs(model.predict(X, operator=pde))[:, 0]
        err_eq = torch.tensor(Y)
        X_ids = torch.topk(err_eq, NumDomain//200, dim=0)[1].numpy()
        '''
        new_inputs, new_targets = gene.generate_samples(config.Algorithm_type[2], 
                cur_xs_selected, cur_ys_selected, config.ys_weight, None)
        truth_error = predict_model(model, new_inputs, new_targets, train_dataset.device, torch.float64).astype(np.float64)
        err_eq = torch.tensor(truth_error)

        X_ids = torch.topk(err_eq, config.Algorithm_type[3], dim=0)[1].numpy()

        save_inputs = np.array(new_inputs)[X_ids].tolist()
        save_targets = np.array(new_targets)[X_ids].tolist()
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
