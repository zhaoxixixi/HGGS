import sys
sys.path.extend(['../../../../../'])
import torch
import os
from yu.tasks.OnlineSamplingTools.dataloader import FileDataSet
from yu.tasks.pde_models import transform
from yu.tasks.OnlineSamplingTools.Baseline_Training_code.Training_epoch import training_epochs

import numpy as np

def predict_model(model, unlabel_loader, save_count: int):
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

    sort_indices = np.argsort(gn)
    save_indices = sort_indices[-save_count:]
    left_indices = sort_indices[:-save_count]

    left_training_indices = np.array(training_indices)[left_indices]
    unlabel_loader.dataset.training_indices = left_training_indices.tolist()

    inputs = np.array(inputs)[save_indices].tolist()
    targets = np.array(targets)[save_indices].tolist()

    return inputs, targets

# Gradient-based Filtering Stage 2 [Uncertainty Sampling - Pool based]
def Ada_Gradient_Init(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch: int=1,
          unlabel_data_path: str=None, save_count: int=None, iters_count: int=None):
    if unlabel_data_path is None:
        assert False, 'Error need unlabel data path!'
    if save_count is None:
        assert False, 'Error need save_count!'
    if iters_count is None:
        assert False, 'Error need iters_count!'

    unlabel_dataset = FileDataSet(
        transform=transform,
        device=train_loader.dataset.device,
        xs_selected=config.xs_selected,
        ys_weight=config.ys_weight,
        norm_xs=config.xs_weight,
        ys_selected=config.ys_selected,
        flag='train',
        *[unlabel_data_path],
        model_name=config.ode_model_name,
    )
    unlabel_loader = torch.utils.data.DataLoader(
        dataset=unlabel_dataset,
        batch_size=len(unlabel_dataset),
        shuffle=True,
    )

    # ----------------------------------train begin----------------------------------
    end_epoch = config.last_epoch_n + begin_epoch

    cur_save_dir = os.path.join(config.save_dir, "Ada-Init-" + str(begin_epoch))
    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration, iteration_test, best_model_loss, False,
          begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping, save_path=cur_save_dir)
    end_epoch = begin_epoch + config.epoch_n

    if save_count == 0:
        with open(os.path.join(config.save_dir, 'Ada-Gradient-data.csv'), 'w+') as f:
            lines = []
            for i in range(len(train_loader.dataset._inputs)):
                line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)
        return begin_epoch, iteration, iteration_test, best_model_loss
    
    cur_xs_selected = []
    cur_ys_selected = []
    for i in range(len(config.ys_selected)):
        cur_ys_selected.append(4 * (i + 1))

    for i in range(len(config.xs_selected)):
        cur_xs_selected.append(config.param_selected[i] + cur_ys_selected[-1] + 1)

    train_dataset = train_loader.dataset
    for _ in range(iters_count):
        new_inputs, new_targets = predict_model(model, unlabel_loader, save_count)
        # add new data
        train_dataset.residule_new_data(new_inputs, new_targets)

        cur_save_dir = os.path.join(config.save_dir, "Ada-Init-" + str(begin_epoch))
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        
        with open(os.path.join(cur_save_dir, 'residual_data.csv'), 'w+') as f:
            lines = []
            for i in range(len(new_inputs)):
                line = ",".join([str(_) for _ in (new_inputs[i] + new_targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)

        iteration, iteration_test, last_model_loss, begin_epoch = training_epochs(config,
            model, optimizer, scheduler, train_loader, test_loader,
            iteration, iteration_test, best_model_loss, False,
            begin_epoch, end_epoch, Early_Stopping=config.Early_Stopping, save_path=cur_save_dir)
        end_epoch = begin_epoch + config.epoch_n

    with open(os.path.join(config.save_dir, 'Ada-Gradient-data.csv'), 'w+') as f:
        lines = []
        for i in range(len(train_loader.dataset._inputs)):
            line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
            lines.append(line)

        f.writelines(lines)

    return begin_epoch, iteration, iteration_test, best_model_loss