import os
import torch
import numpy as np
import sys
sys.path.extend(['../../../../../'])
from yu.core.logger import logger
from yu.nn.EarlyStopping import EarlyStopping
from yu.tasks.OnlineSamplingTools.Baseline_Training_code.Training_epoch import training_epoch, training_epochs

def IS_dag_training(config,
                 model, optimizer, scheduler, train_loader, test_loader,
                 iteration=0, iteration_test=0, best_model_loss=1e9):
    global_early_stopping = EarlyStopping(save_path=config.save_dir, patience=5)

    # Initialized Training Begin
    end_epoch = config.last_epoch_n + 0
    iteration, iteration_test, model_final_loss, last_epoch_training = training_epochs(config,
                 model, optimizer, scheduler,
                 train_loader, test_loader,
                 iteration, iteration_test, best_model_loss, True,
                 begin_epoch=0, end_epoch=end_epoch,
                 Early_Stopping=config.Early_Stopping)
    # Initialized Training End

    # IS Finetune Begin
    model_loss = []
    last_finetune_epoch = last_epoch_training
    for finetune_iter in range(config.finetune_count):
        if config.tau == -1:
            cur_tau = (finetune_iter + 1) / config.finetune_count * 0.9
        else:
            cur_tau = config.tau

        iteration, iteration_test, cur_model_loss, last_finetune_epoch = IS_finetune(config,
          model, optimizer, scheduler,
          train_loader, test_loader,
          iteration, iteration_test, best_model_loss,
          begin_epoch=last_finetune_epoch,
          tau=cur_tau,
          Early_Stopping=config.Early_Stopping
        )

        model_loss.append(cur_model_loss)

        global_early_stopping(cur_model_loss[-1], model)
        if global_early_stopping.early_stop:
            break

    model_loss = np.array(model_loss)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    x_label = np.arange(1, len(model_loss) + 1)
    plt.plot(x_label, model_loss[:, 0].tolist(), label='Best Loss')
    plt.plot(x_label, model_loss[:, 1].tolist(), label='Final Loss')

    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.gca().yaxis.set_major_formatter('{:.5f}'.format)
    plt.xticks(range(1, len(model_loss) + 1))
    plt.grid(True)

    plt.legend()
    plt.savefig(os.path.join(config.save_dir, 'finetune-loss.png'))
    plt.close()
    # IS Finetune End

    with open(os.path.join(config.save_dir, 'data.csv'), 'w+') as f:
        lines = []
        for i in range(len(train_loader.dataset._inputs)):
            line = ",".join(
                [str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
            lines.append(line)

        f.writelines(lines)


def IS_finetune(config,
             model, optimizer, scheduler, train_loader, test_loader,
             iteration=0, iteration_test=0, best_model_loss=1e9,
             begin_epoch: int = 1, tau: float = None, Early_Stopping: bool = False):
    model_best_epoch = 0
    save_cur_path = os.path.join(config.save_dir, './finetune/{}'.format(begin_epoch))
    if not os.path.exists(save_cur_path):
        os.makedirs(save_cur_path)

    # Early Stopping
    if Early_Stopping:
        early_stopping = EarlyStopping(save_cur_path, patience=300)

    ################## calculate the weight of samples for resampling [Begin]
    train_data = train_loader.dataset
    model.eval()

    grad_list = []
    indices_list = []
    for i, (inputs, targets, _indices) in enumerate(train_loader):
        outputs = model(inputs)
        with torch.no_grad():
            grad_norm = (torch.norm(2 * (outputs - targets), dim=-1).detach().cpu().numpy())  # grad_norm
        grad_list.extend(grad_norm)
        indices_list.extend(_indices.cpu().numpy())  # record the indices of train samples

    sample_grad_norm = np.zeros(len(train_data.initial_indices))
    for _idx, _grad_norm in zip(indices_list, grad_list):
        if sample_grad_norm[_idx] != 0:
            sample_grad_norm[_idx] = np.mean(
                [sample_grad_norm[_idx], _grad_norm])  # (sample_grad_norm[_idx] + _grad_norm) / 2
        else:
            # print(_grad_norm)
            sample_grad_norm[_idx] = _grad_norm
    sum_grad_norm = np.sum(sample_grad_norm)
    cur_prob = sample_grad_norm / sum_grad_norm  # normalization probability(like the scores)

    # resampling probs
    new_prob = cur_prob * tau + (1 - tau) * train_data.initial_prob
    finetune_indices = np.random.choice(train_data.initial_indices, size=len(train_data), replace=True, p=new_prob)
    # replace training data
    train_data.update_train_indices(finetune_indices)
    ################## calculate the weight of samples for resampling [End]

    last_epoch = begin_epoch
    # save training resampled data
    with open(os.path.join(save_cur_path, 'finetune-data.csv'), 'w+') as f:
        lines = []
        for i, (inputs, targets, _indices) in enumerate(train_loader):

            for j in _indices:
                fields = []
                fields += train_data._inputs[j]
                fields += train_data._targets[j]
                line = ",".join([str(_) for _ in fields]) + '\n'
                lines.append(line)

        f.writelines(lines)

    # save training samples indices
    with open(os.path.join(save_cur_path, 'finetune-data-indices.csv'), 'w+') as f:
        lines = []
        for i, (inputs, targets, _indices) in enumerate(train_loader):

            for j in _indices:
                line = str(j.item()) + '\n'
                lines.append(line)

        f.writelines(lines)

    last_model_loss = None
    # ----------------------------------train begin----------------------------------
    for epoch in range(begin_epoch, config.finetune_epoch + begin_epoch):
        iteration, iteration_test, best_model_loss, last_model_loss, model_best_epoch = \
            training_epoch(config, model, optimizer, scheduler, train_loader, test_loader,
                   iteration, iteration_test, best_model_loss, True,
                   epoch, save_cur_path, model_best_epoch=model_best_epoch)
        last_epoch = epoch + 1

        if early_stopping:
            early_stopping(last_model_loss, model)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
    model.save(os.path.join(save_cur_path, f'model_final.pth'))
    # ----------------------------------train end----------------------------------

    with open(os.path.join(save_cur_path, 'model_loss.txt'), 'w+') as f:
        f.write('model_best Testing Loss: {%.8f}\n' % (best_model_loss))
        f.write('model_final Testing Loss: {%.8f}\n' % (last_model_loss))
        f.write('model_best epoch: {%d}\n' % (model_best_epoch))

    # back to previous random distribution
    train_data.update_train_indices()
    return iteration, iteration_test, [best_model_loss, last_model_loss], last_epoch
