import sys
sys.path.extend(['../../../../../'])
from yu.core.logger import logger
import torch
import numpy as np
import wandb
import os
from yu.nn.EarlyStopping import EarlyStopping

def train_IS(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9,
          begin_epoch: int=1,
          save: bool=False, Early_Stopping: bool=False):
    if Early_Stopping:
        early_stopping = EarlyStopping(save_path=config.save_dir, patience=300)
    else:
        early_stopping = None

    model_best_epoch = 0
    last_model_loss = None
    # ----------------------------------train begin----------------------------------
    cur_training_path = os.path.join(config.save_dir, str(begin_epoch))
    if not os.path.exists(cur_training_path):
        os.makedirs(cur_training_path)

    end_epoch = config.last_epoch_n + begin_epoch
    for epoch in range(begin_epoch, end_epoch):
        if epoch - begin_epoch < config.warm_up_epoch:
            cur_tau = 0.0
        elif config.tau == -1:
            cur_tau = (epoch - begin_epoch) / (end_epoch - begin_epoch)
        else:
            cur_tau = config.tau

        logger.info(f'Starting epoch {epoch}')
        model.train()

        cur_train_loss = 0
        cur_train_samples = 0


        train_grad_norm = []  # grad norm to update the IS
        train_sample_indices = []

        for i, (inputs, targets, _indices) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = config.loss_func(outputs, targets)

            with torch.no_grad():
                grad_norm = (torch.norm(2 * (outputs - targets), dim = -1).detach().cpu().numpy())  # grad_norm

            loss.backward()

            optimizer.step()

            train_grad_norm.extend(grad_norm)
            train_sample_indices.extend(_indices.cpu().numpy())  # record the indices of train samples

            cur_train_loss += loss.item() * len(inputs)
            cur_train_samples += len(inputs)
            logger.info(
                f'gpu {config.gpu_idx} Training epoch {epoch} iteration {iteration} has finished, LOSS: {loss:.8f}')
            iteration += 1
        
        cur_train_loss /= cur_train_samples
        if config.wandb_swith:
            wandb.log({"Training loss": cur_train_loss}, step=epoch)
        if config.wandb_swith:
            wandb.log({'lr': scheduler.get_lr()[0] if scheduler else optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch)

        if scheduler:
            scheduler.step()
        # test data
        logger.info('*' * 20)
        logger.info(f'Starting testing, epoch {epoch}')
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_inputs_size = 0
            for i, (inputs, targets, _indices) in enumerate(test_loader):

                outputs = model(inputs)

                loss = config.loss_func(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                test_inputs_size += inputs.size(0)

            logger.info(
                f'gpu {config.gpu_idx} Testing epoch {epoch} iteration {iteration_test} has finished,'
                f' LOSS: {test_loss / test_inputs_size:.8f}'
            )
            if config.wandb_swith:
                wandb.log({"Testing loss": test_loss / test_inputs_size}, step=epoch)
            iteration_test += 1
            if test_loss / test_inputs_size < best_model_loss:
                model_best_epoch = epoch
                best_model_loss = test_loss / test_inputs_size
                model.save(os.path.join(config.save_dir, f'model_best.pth'))
                model.save(os.path.join(cur_training_path, f'model_best.pth'))
            last_model_loss = test_loss / test_inputs_size\
        
        if early_stopping:
            early_stopping(last_model_loss, model)
            if early_stopping.early_stop:
                with open(os.path.join(cur_training_path, 'EarlyStop.txt'), 'w+') as f:
                    f.write("Early Stopping in epoch {}\n".format(epoch))
                break
        if cur_tau == 0.0:
            continue
        train_data = train_loader.dataset
        sample_grad_norm = np.zeros(len(train_data.initial_indices))
        for _idx, _grad_norm in zip(train_sample_indices, train_grad_norm):
            if sample_grad_norm[_idx] != 0:
                sample_grad_norm[_idx] = np.mean([sample_grad_norm[_idx], _grad_norm])
            else:
                sample_grad_norm[_idx] = _grad_norm
        sum_grad_norm = np.sum(sample_grad_norm)
        cur_prob = sample_grad_norm / sum_grad_norm  # normalization probability(like the scores)
        new_prob = cur_prob * cur_tau + (1 - cur_tau) * train_data.initial_prob
        indices_next_epoch = np.random.choice(train_data.initial_indices, size=len(train_data), replace=True, p=new_prob)
        train_data.update_train_indices(indices_next_epoch)

    model.save(os.path.join(config.save_dir, f'model_final.pth'))
    model.save(os.path.join(cur_training_path, f'model_final.pth'))
    # ----------------------------------train end----------------------------------
    # save loss result
    with open(os.path.join(config.save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best Testing Loss: {%.8f}\n' % (best_model_loss))
        f.write('model_final Testing Loss: {%.8f}\n' % (last_model_loss))
        f.write('model_best epoch: {%d}\n' % (model_best_epoch))

    with open(os.path.join(cur_training_path, 'model_loss.txt'), 'w+') as f:
        f.write('model_best Testing Loss: {%.8f}\n' % (best_model_loss))
        f.write('model_final Testing Loss: {%.8f}\n' % (last_model_loss))
        f.write('model_best epoch: {%d}\n' % (model_best_epoch))

    if save:
        with open(os.path.join(config.save_dir, 'data.csv'), 'w+') as f:
            lines = []
            for i in range(len(train_loader.dataset._inputs)):
                line = ",".join([str(_) for _ in (train_loader.dataset._inputs[i] + train_loader.dataset._targets[i])]) + '\n'
                lines.append(line)

            f.writelines(lines)
    return iteration, iteration_test
