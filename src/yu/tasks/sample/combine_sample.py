import os
from multiprocessing import  Process
import multiprocessing
import sys

sys.path.extend(['../../../../src'])    # noqa
from yu.core.logger import logger
from yu.tasks.sample.generate_data import score_calculator
from yu.tools.grid import Grid
from yu.tools.misc import makedir
import time


path = '../../../../resource/sample/generate_data.env'

def sample(data_path, core_idx, config, truth, _y0, _T, _T_unit):
    root_dir = config.save_dir + '/{}sub/{}'.format(config.core, core_idx)
    makedir(root_dir)
    good_f = open(os.path.join(root_dir, 'good.csv'), 'a+')
    bad_f = open(os.path.join(root_dir, 'bad.csv'), 'a+')

    t1 = time.time()
    grid = Grid(
        max_split_times=config.max_split_times,
        per_split_n=config.per_split_n,
        init_split_n=config.init_split_n,
        dimensions=len(config.params_selected),
        threshold=config.threshold,
        save_dir=root_dir,
        search_type=config.search_type,
        db_path=config.db_path,
        table=config.tablename,
    )
    if os.path.exists(grid.config_file):
        grid.reload(root_dir, grid)
    grid.execute(
        score_calculator=score_calculator,
        config=config,
        truth=truth,
        good_f=good_f,
        bad_f=bad_f,
        t1=time.time(),
        data_path=data_path,
        reload_path=os.path.join(config.reload_path, str(core_idx) + '.txt') if config.reload else None,
        _Parameters=truth._Parameters,
        _y0=_y0, _T=_T, _T_unit=_T_unit,
    )
    t2 = time.time()
    logger.info(
        f'total: {config.n_good + config.n_bad}/{config.total_good + config.total_bad},'
        f' n_good: {config.n_good}/{config.total_good}, n_bad: {config.n_bad}/{config.total_good},'
        f' running time: {t2 - t1}s'
    )
    good_f.close()
    bad_f.close()

def run_(data_path, core, config, truth, _y0, _T, _T_unit):
    process_list = []
    for i in range(core):
        core_data_path = os.path.join(data_path, '{}.txt'.format(i))
        p = Process(target=sample, args=(core_data_path, i, config, truth, _y0, _T, _T_unit))  # 实例化进程对象
        process_list.append(p)
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print('subData done!')
    print('Combine All the Data')

    good_f = open(config.good_f, 'a+')
    bad_f = open(config.bad_f, 'a+')
    for i in range(core):
        root_dir = config.save_dir + '/{}sub/{}'.format(config.core, i)
        good_data_path = os.path.join(root_dir, 'good.csv')
        bad_data_path = os.path.join(root_dir, 'bad.csv')

        with open(good_data_path, 'r') as f:
            good_f.writelines(f.readlines())
        with open(bad_data_path, 'r') as f:
            bad_f.writelines(f.readlines())
    good_f.close()
    bad_f.close()

if __name__ == '__main__':
    run_()
    print('end')