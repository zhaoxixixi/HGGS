import sys
from .Training_epoch import training_epochs
sys.path.extend(['../../../../../'])
from yu.core.logger import logger
import torch
import os
from yu.tasks.OnlineSamplingTools.dataloader import FileDataSet
from yu.tasks.pde_models import transform

import wandb
import numpy as np

def weightedRS(model, unlabel_loader, train_dataset, save_count: int, weighted_sum: float):
    '''
    return replace_count, weighted_sum
    '''
    inputs = []
    targets = []
    gn = []
    training_indices = []
    model.eval()

    with torch.no_grad():
        for (input, target, _idx) in unlabel_loader:
            inputs.extend(np.array(unlabel_loader.dataset._inputs)[_idx].tolist())
            targets.extend(np.array(unlabel_loader.dataset._targets)[_idx].tolist())

            output = model(input)

            grad_norm = (torch.norm(2 * (output - target), dim=-1).detach().cpu().numpy())  # grad_norm
            gn.extend(grad_norm)
            training_indices.extend(_idx)
    
    if len(inputs) < save_count:
        save_count = len(inputs)

    # random sampling the candidate unlabeled samples    
    RS_random_indices = np.random.choice(len(training_indices), size=save_count, replace=False)

    # iters each samples & train_dataset
    M_training_set = len(train_dataset)
    candidate_change_train_dataset_indices = np.random.choice(len(train_dataset), size=len(RS_random_indices), replace=False)
    replace_count = 0
    cur_save_training_indices = []

    for i in range(len(RS_random_indices)):
        cur_idx = RS_random_indices[i]

        weighted_sum = weighted_sum + gn[cur_idx]
        sampling_prob = min(gn[cur_idx] / weighted_sum, 1.0 / M_training_set)

        skip_prob = 1 - M_training_set * sampling_prob
        _r = np.random.uniform(0, 1)
        if _r < skip_prob:
            continue

        # change the candidate sample input&target
        cur_candidate_idx = train_dataset.training_indices[candidate_change_train_dataset_indices[i]]
        train_dataset._inputs[cur_candidate_idx] = inputs[cur_idx]
        train_dataset._targets[cur_candidate_idx] = targets[cur_idx]

        replace_count += 1
        cur_save_training_indices.append(training_indices[cur_idx])

    # reset inputs&targets in memory
    train_dataset.init_inputs_targets()

    # save left unlabeled training dataset
    left_training_indices = np.setdiff1d(training_indices, cur_save_training_indices).tolist()
    unlabel_loader.dataset.training_indices = left_training_indices
    
    return replace_count, weighted_sum


def WeightedReservoirSampling(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch: int=1,
          unlabel_data_path: str=None, save_count: int=None):
    if unlabel_data_path is None:
        assert False, 'Error need unlabel data path!'
    if save_count is None:
        assert False, 'Error need save_count!'

    unlabel_dataset = FileDataSet(
        transform=transform,
        device=train_loader.dataset.device,
        xs_selected=config.xs_selected,
        ys_weight=config.ys_weight,
        # xs_weigth=config.xs_weight,
        norm_xs=config.xs_weight,
        ys_selected=config.ys_selected,
        flag='train',
        *[unlabel_data_path],
        model_name=config.ode_model_name,
    )
    unlabel_loader = torch.utils.data.DataLoader(
        dataset=unlabel_dataset,
        batch_size=len(unlabel_dataset),
        shuffle=False,
    )
    from yu.nn.EarlyStopping import EarlyStopping
    if config.Early_Stopping:
        early_stopping = EarlyStopping(save_path=config.save_dir, patience=5)
    else:
        early_stopping = None


    # ----------------------------------train begin----------------------------------
    end_epoch = config.last_epoch_n + begin_epoch
    iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration, iteration_test, best_model_loss, True,
          begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping)
    end_epoch = begin_epoch + config.epoch_n

    early_stopping(last_model_loss, model)

    cur_xs_selected = []
    cur_ys_selected = []
    for i in range(len(config.ys_selected)):
        cur_ys_selected.append(4 * (i + 1))

    for i in range(len(config.xs_selected)):
        cur_xs_selected.append(config.param_selected[i] + cur_ys_selected[-1] + 1)

    train_dataset = train_loader.dataset
    weighted_sum = 0.0
    for _ in range(config.iter_count):
        # Reservoir Sampling sampels
        replace_count, weighted_sum = weightedRS(model, unlabel_loader, train_dataset, save_count, weighted_sum)

        cur_save_dir = os.path.join(config.save_dir, str(begin_epoch))
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        
        with open(os.path.join(cur_save_dir, 'cur_training_dataset.csv'), 'w+') as f:
            lines = []
            for i in range(len(train_dataset._inputs)):
                line = ",".join([str(_) for _ in (train_dataset._inputs[i] + train_dataset._targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)
        
        with open(os.path.join(cur_save_dir, 'left_unlabeled_dataset.csv'), 'w+') as f:
            lines = []
            for i in unlabel_dataset.training_indices:
                line = ",".join([str(_) for _ in (unlabel_dataset._inputs[i] + unlabel_dataset._targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)

        with open(os.path.join(cur_save_dir, 'replace_count.txt'), 'w+') as f:
            f.write('replace_cur_training: {}\n'.format(replace_count))

        iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
            model, optimizer, scheduler, train_loader, test_loader,
            iteration, iteration_test, best_model_loss, True,
            begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping)
        end_epoch = begin_epoch + config.epoch_n

        if early_stopping:
            early_stopping(last_model_loss, model)
            if early_stopping.early_stop:
                with open(os.path.join(config.save_dir, 'EarlyStop.txt'), 'w+') as f:
                    f.write("Early Stopping in fintune_iters {}\n".format(_))
                break

    with open(os.path.join(config.save_dir, 'data.csv'), 'w+') as f:
        lines = []
        for i in range(len(train_loader.dataset._inputs)):
            line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
            lines.append(line)

        f.writelines(lines)
