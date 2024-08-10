'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-21 03:19:45
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-13 12:56:16
FilePath: /rh/Sample/src/yu/tools/pre_produce_data.py
'''
import os.path

import numpy as np
import sys

sys.path.extend(['../../../../src'])    # noqa
from yu.tools.misc import makedir


cnt = 0
idx = -1
f = []

def produce_data(config):
    '''
    split total data to grid
    '''

    # params_selected = [0, 1, 2, 3, 4]
    # log_lb = 0
    # log_ub = 10
    params_selected = config.params_selected
    log_lb = config.log_lb
    log_ub = config.log_ub
    # path = r'../../../../data/{}param/all.csv'.format(len(params_selected))

    # edge = 15  # per_split_n
    # cube_n = 5
    edge = config.per_split_n[0]
    cube_n = config.cube_n[0]

    scale = (log_ub - log_lb) / edge

    # cpu number: core
    core = config.core
    data_path = config.data_path
    # save path xx/{num}param/sub{core}/y.txt
    save_path = data_path.format(len(params_selected), core)
    makedir(save_path)

    N = edge ** len(params_selected)
    sub_N = (N) // core

    def dfs(now, edge_n, edge_l):
        global cnt, idx, f
        if len(now) == edge_l:

            if cnt == 0:
                if len(f) == core:
                    import random
                    f[random.randint(0, core - 1)].write(str(now) + '\n')
                    return 
                cnt = sub_N
                idx += 1
                # if len(f) != 0:
                #     f[-1].close()
                f2 = open(os.path.join(save_path, "{}.txt".format(idx)), 'w+')
                f.append(f2)

            f[-1].write(str(now) + '\n')
            cnt -= 1
            return

        now.append(-1)
        for i in range(edge_n):
            now[-1] = i
            dfs(now, edge_n, edge_l)
        now.pop()

    dfs([], edge, len(params_selected))
    for f1 in f:
        f1.close()
# print(count)