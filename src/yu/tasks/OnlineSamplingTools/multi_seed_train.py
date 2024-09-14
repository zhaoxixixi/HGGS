import time
from multiprocessing import Process
import multiprocessing

import sys

sys.path.extend(['../../../../src'])  # noqa

import sys, os
lock = multiprocessing.Lock()
from yu.tasks.BioSysConfig.BioSysConfig import SysConfig
from yu.tools.misc import get_sampling_name
def modify(path, seeds):
    if os.path.isfile(path):
        return
    dir_list = os.listdir(path)
    ok_dir_list = []
    for dir in dir_list:
        if '2024' not in dir:
            continue
        ok_dir_list.append(dir)
    # print(ok_dir_list)
    flag = len(ok_dir_list) == len(seeds)
    print(path, flag)
    if flag:
        dir_list.sort()
        idx = 0
        for dir in dir_list:
            if '2024' in dir:
                dir_list[idx] = dir
                idx += 1
        dir_list = dir_list[:idx]
        for i in range(len(dir_list)):
            old = os.path.join(path, dir_list[i])
            new = os.path.join(path, str(seeds[i % len(seeds)]))
            os.rename(old, new)
        return
    else:
        for dir_name in dir_list:
            modify(os.path.join(path, dir_name), seeds)

path = '../../../../resource/OnlineSamplingTools/train.env'

def train(cmd_string):
    """
    :param cmd_string: cmd command: 'adb devices'
    :return:
    """
    import subprocess

    print('run: {}'.format(cmd_string))
    return subprocess.Popen(cmd_string, shell=True, stdout=None, stderr=None).wait()


save_path = None
def train_fuc(x, y, seed, train_set, data_path, model_path, gpu,
              Algorithm_type: str = "['gene']", iter_count: int = 15,
              epoch_n: int = 1000, warm_up_epoch: int = 100,
              dataset_type: str = 'default', boundary_sampling_ratio: float = 0.05, boundary_KNN: int = 10,
              finetune_epoch: int = 500, finetune_count: int = 20,
              last_epoch_n: int = 1000, base_lr: float = 1e-6,
              train_strategy: str = "Point-GN",
              uniform_sampling_ratio: float = 0.05,
              ode_model_name: str = 'HSECC',
              test_paths: str = "../../../../data/6param/test/new_test/test_5k.csv",
              xs_param: list = [1.53, 0.04, 1.35, 0.02, 1.35, 0.1, 0.00741], param_selected: list = [0, 1, 2, 3, 4, 5],
              max_lr: float = 0.0001,
              pretrained_model_path: str = None,
              nn_layers: list[int] = [128, 128, 128, 128],
              nn_norm: str = '["BatchNorm1d","BatchNorm1d","BatchNorm1d","BatchNorm1d"]',
              tau: float = 0.5,
              lr_alpha: float = 0.2, total_training_samples: int = 5000,
              dropout: float = 0.2, batch_size: int = 40960,
              xs_lb_ub: list[int] = [0, 10], Ada_Gradient_Settings: list[any] = ["Path", 5],
              Gaussian_Mixture: bool = False):
    global path
    # lock.acquire()
    # lock2.acquire()
    try:
        ans = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if 'seed=' in line[:len("seed=")]:
                    ans.append('seed={}\n'.format(seed))
                elif 'save_dir' in line:
                    ans.append(
                        "save_dir=\"../../../../output/mlp/reg_2/{}param/{}/\"\n".format(len(x), model_path))
                elif 'log_dir' in line:
                    ans.append("log_dir=\"/home/users/rh/Sample/tf_logs/{}param/reg_2/\"\n".format(len(x)))
                elif 'train_paths' in line[:len('train_paths')]:
                    ans.append(
                        "train_paths=[\"../../../../data/{}param/{}/{}.csv\"]\n".format(len(x), data_path, train_set))
                elif 'xs_selected' in line:
                    ans.append("xs_selected={}\n".format(str(x)))
                elif 'ys_selected' in line:
                    ans.append("ys_selected={}\n".format(str(y)))
                elif 'name' in line and 'ys_name' not in line and 'ode_model_name' not in line:
                    ans.append("name=\"{}\"\n".format(train_set))

                elif 'ode_model_name' in line:
                    ans.append("ode_model_name=\"{}\"\n".format(ode_model_name))
                elif 'test_paths' in line:
                    ans.append("test_paths=[\"{}\"]\n".format(test_paths))
                elif 'xs_param' in line:
                    ans.append("xs_param={}\n".format(xs_param))
                elif 'param_selected' in line:
                    ans.append("param_selected={}\n".format(param_selected))

                elif 'gpu_idx' in line:
                    if type(gpu) is str:
                        ans.append("gpu_idx=\"cpu\"\n")
                    else:
                        ans.append("gpu_idx={}\n".format(gpu))

                elif 'iter_count' in line:
                    ans.append("iter_count={}\n".format(iter_count))
                elif 'Algorithm_type' in line:
                    ans.append("Algorithm_type={}\n".format(Algorithm_type))
                elif 'finetune_epoch' in line:
                    ans.append("finetune_epoch={}\n".format(finetune_epoch))
                elif 'finetune_count' in line:
                    ans.append("finetune_count={}\n".format(finetune_count))
                elif 'base_lr' in line:
                    ans.append("base_lr={}\n".format(base_lr))
                elif 'max_lr' in line:
                    ans.append("max_lr={}\n".format(max_lr))

                elif 'epoch_n' in line and 'last_epoch_n' not in line:
                    ans.append("epoch_n={}\n".format(epoch_n))
                elif 'last_epoch_n' in line:
                    ans.append("last_epoch_n={}\n".format(last_epoch_n))
                elif 'warm_up_epoch' in line:
                    ans.append("warm_up_epoch={}\n".format(warm_up_epoch))
                elif 'boundary_KNN' in line:
                    ans.append("boundary_KNN={}\n".format(boundary_KNN))

                elif 'dataset_type' in line:
                    ans.append("dataset_type={}\n".format(dataset_type))
                elif 'boundary_sampling_ratio' in line:
                    ans.append("boundary_sampling_ratio={}\n".format(boundary_sampling_ratio))
                elif 'uniform_sampling_ratio' in line:
                    ans.append("uniform_sampling_ratio={}\n".format(uniform_sampling_ratio))
                elif 'boundary_KNN' in line:
                    ans.append("boundary_KNN={}\n".format(boundary_KNN))

                elif 'train_strategy' in line:
                    ans.append("train_strategy={}\n".format(train_strategy))

                elif 'pretrained_model_path' in line:
                    ans.append("pretrained_model_path={}\n".format(pretrained_model_path))


                elif 'nn_layers' in line:
                    ans.append("nn_layers={}\n".format(nn_layers))
                elif 'nn_norm' in line:
                    ans.append("nn_norm={}\n".format(nn_norm))

                elif 'tau' in line:
                    ans.append("tau={}\n".format(tau))

                # lr_alpha
                elif 'lr_alpha' in line:
                    ans.append("lr_alpha={}\n".format(lr_alpha))
                elif 'total_training_samples' in line:
                    ans.append("total_training_samples={}\n".format(total_training_samples))
                elif 'dropout' in line:
                    ans.append("dropout={}\n".format(dropout))

                elif 'batch_size' in line:
                    ans.append("batch_size={}\n".format(batch_size))

                # xs_lb_ub
                elif 'xs_lb_ub' in line:
                    ans.append("xs_lb_ub={}\n".format(xs_lb_ub))
                # Ada_Gradient_Settings
                elif 'Ada_Gradient_Settings' in line:
                    ans.append("Ada_Gradient_Settings={}\n".format(Ada_Gradient_Settings))
                # Gaussian_Mixture
                elif 'Gaussian_Mixture' in line:
                    ans.append("Gaussian_Mixture={}\n".format(Gaussian_Mixture))

                else:
                    ans.append(line)
        with open(path, 'w+') as f:
            f.writelines(ans)

        with open(path, 'r') as f:
            print(f.readlines())
        com_str = 'python train.py'

        train(com_str)
    finally:
        pass


'''
multi_seed script
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_set')  #
parser.add_argument('--model_path', type=str, default='')  # model save path
parser.add_argument('--seeds', type=str, default="[53, 25, 99, 81, 50]")  # training with seeds
parser.add_argument('--ode_model_name', type=str, default="HSECC")  # system model path

parser.add_argument('--epoch_n', type=str, default="1000")  # epoch
parser.add_argument('--warm_up_epoch', type=str, default="100")  # warm up epoch
parser.add_argument('--last_epoch_n', type=str, default="1000")  # first training epoch
parser.add_argument('--base_lr', type=str, default="1e-6")  # learning rate
parser.add_argument('--total_training_samples', type=str, default="5000")  # total sampling count
parser.add_argument('--batch_size', type=str, default="40960")  # batch size

args = parser.parse_args()

gpu_list = [0, 0, 0, 0, 0]
pool = multiprocessing.Pool(processes=len(gpu_list))
if __name__ == '__main__':
    sampling_name = get_sampling_name(args.train_set)
    system_config = SysConfig(sampling_name, args.ode_model_name)

    Ada_Gradient_Settings = system_config.Ada_Gradient_Settings
    Gaussian_Mixture = system_config.Gaussian_Mixture
    tau = system_config.tau
    train_strategy = system_config.train_strategy
    # dataset settings
    dataset_type = system_config.dataset_type
    boundary_sampling_ratio = system_config.boundary_sampling_ratio
    uniform_sampling_ratio = system_config.uniform_sampling_ratio
    boundary_KNN = system_config.boundary_KNN
    # Importance Sampling settings
    finetune_epoch = system_config.finetune_epoch
    finetune_count = system_config.finetune_count
    # system model settings
    ode_model_name = system_config.ode_model_name
    param_selected = system_config.param_selected
    xs_selected = system_config.xs
    ys_selected = system_config.ys
    xs_param = system_config.xs_param
    xs_lb_ub = system_config.xs_lb_ub
    # testing set
    test_paths = system_config.test_path
    # training set
    train_set = args.train_set
    data_path = system_config.data_path
    model_path = args.model_path    # model save path
    # sampling settings
    Algorithm_type = system_config.Algorithm_type
    iter_count = system_config.iter_count

    # training model settings
    batch_size = eval(args.batch_size)
    total_training_samples = eval(args.total_training_samples)
    # max_lr = float(args.max_lr)
    epoch_n = int(args.epoch_n)
    last_epoch_n = int(args.last_epoch_n)
    warm_up_epoch = int(args.warm_up_epoch)
    seeds = eval(args.seeds)
    # model structure and learning parameters
    dropout = system_config.dropout
    nn_layers = system_config.nn_layers
    nn_norm = system_config.nn_norm
    lr_alpha = system_config.lr_alpha
    base_lr = float(args.base_lr)

    process_list = []
    for i in range(len(seeds)):
        p = Process(target=train_fuc, args=(xs_selected, ys_selected, seeds[i], train_set, data_path,
                                            model_path, gpu_list[i % len(gpu_list)],
                                            Algorithm_type, iter_count,
                                            epoch_n, warm_up_epoch,
                                            dataset_type, boundary_sampling_ratio, boundary_KNN,
                                            finetune_epoch, finetune_count,
                                            last_epoch_n, base_lr,
                                            train_strategy,
                                            uniform_sampling_ratio,
                                            ode_model_name,
                                            test_paths,
                                            xs_param, param_selected,
                                            -1,
                                            None,
                                            nn_layers,
                                            nn_norm, tau,
                                            lr_alpha, total_training_samples,
                                            dropout,
                                            batch_size,
                                            xs_lb_ub, Ada_Gradient_Settings, Gaussian_Mixture))
        process_list.append(p)
        
    for i in range((len(process_list) + len(gpu_list) - 1) // (len(gpu_list))):
        Max = min(len(process_list), (i + 1) * len(gpu_list))
        for j in range(i * len(gpu_list), Max):
            # print(j, i, gpu_list[j % len(gpu_list)])
            # continue
            process_list[j].start()
            import random

            time.sleep(random.randint(15, 30))
            # time.sleep(180)
        for j in range(i * len(gpu_list), Max):
            process_list[j].join()

    # train done
    print('train End')
    # rename

    save_path = '../../../../output/mlp/reg_2/{}param/{}/{train_set}/'.format(len(xs_selected), model_path,
                                                                              train_set=train_set)
    modify(save_path, seeds)