import sys
sys.path.extend(['../../../../../'])
from yu.core.logger import logger
import torch
import os
import wandb

# training for one epoch
def training_epoch(config, model, optimizer, scheduler, train_loader, test_loader,
                   iteration=0, iteration_test=0, best_model_loss=1e9, log=True,
                   epoch: int=-1, cur_save_dir: str=None, model_best_epoch: int=-1):
    logger.info(f'Starting epoch {epoch}')
    model.train()

    cur_train_loss = 0
    cur_train_samples = 0

    for i, (inputs, targets, _indices) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = config.loss_func(outputs, targets)

        loss.backward()

        optimizer.step()

        cur_train_loss += loss.item() * len(inputs)
        cur_train_samples += len(inputs)

        # wandb.log({"Training_iteration": iteration})
        logger.info(
            f'gpu {config.gpu_idx} Training epoch {epoch} iteration {iteration} has finished, LOSS: {loss.item():.8f}')
        iteration += 1

    cur_train_loss /= cur_train_samples
    if config.wandb_swith and log:
        wandb.log({"Training loss": cur_train_loss}, step=epoch)
    if config.wandb_swith and log:
        wandb.log({'lr': scheduler.get_lr()[0] if scheduler else optimizer.state_dict()['param_groups'][0]['lr']},
                  step=epoch)

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
        if config.wandb_swith and log:
            wandb.log({"Testing loss": test_loss / test_inputs_size}, step=epoch)
        iteration_test += 1
        if test_loss / test_inputs_size < best_model_loss:
            model_best_epoch = epoch
            best_model_loss = test_loss / test_inputs_size
            model.save(os.path.join(config.save_dir, f'model_best.pth'))
            model.save(os.path.join(cur_save_dir, f'model_best.pth'))
        last_model_loss = test_loss / test_inputs_size

    return iteration, iteration_test, best_model_loss, last_model_loss, model_best_epoch

# training for end_epoch - begin_epoch + 1 epoch with Early Stopping
def training_epochs(config,
          model, optimizer, scheduler, train_loader, test_loader,
          iteration=0, iteration_test=0, best_model_loss=1e9, log=True,
          begin_epoch: int=1, end_epoch: int=1000, Early_Stopping: bool=False, save_path: str=None):
    from yu.nn.EarlyStopping import EarlyStopping

    if save_path:
        cur_save_dir = save_path
    else:
        cur_save_dir = os.path.join(config.save_dir, str(begin_epoch))
    if Early_Stopping:
        early_stopping = EarlyStopping(save_path=cur_save_dir, patience=300)
    else:
        early_stopping = None

    model_best_epoch = 0
    last_model_loss = None

    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)

    cur_end_epoch = begin_epoch
    for epoch in range(begin_epoch, end_epoch):
        iteration, iteration_test, best_model_loss, last_model_loss, model_best_epoch = \
        training_epoch(config, model, optimizer, scheduler, train_loader, test_loader,
                   iteration, iteration_test, best_model_loss, log,
                   epoch, cur_save_dir, model_best_epoch=model_best_epoch)
        cur_end_epoch = epoch + 1
        if early_stopping:
            early_stopping(last_model_loss, model)
            if early_stopping.early_stop:
                with open(os.path.join(cur_save_dir, 'EarlyStop.txt'), 'w+') as f:
                    f.write("Early Stopping in epoch {}\n".format(epoch))
                break

    model.save(os.path.join(config.save_dir, f'model_final.pth'))
    model.save(os.path.join(cur_save_dir, f'model_final.pth'))
    # ----------------------------------train end----------------------------------
    # save loss result
    with open(os.path.join(config.save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best Testing Loss: {%.8f}\n' % (best_model_loss))
        f.write('model_final Testing Loss: {%.8f}\n' % (last_model_loss))
        f.write('model_best epoch: {%d}\n' % (model_best_epoch))

    finetune_save_dir = cur_save_dir
    with open(os.path.join(finetune_save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best Testing Loss: {%.8f}\n' % (best_model_loss))
        f.write('model_final Testing Loss: {%.8f}\n' % (last_model_loss))
        f.write('model_best epoch: {%d}\n' % (model_best_epoch))

    if Early_Stopping:
        return iteration, iteration_test, last_model_loss, cur_end_epoch

    return iteration, iteration_test, None, cur_end_epoch