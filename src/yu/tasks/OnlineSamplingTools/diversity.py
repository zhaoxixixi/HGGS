import os
import pandas as pd
import numpy as np

eps = 1e-6
def get_data_imbalace_ratio(data):
    Oscillation = len(np.where(data > 0.0)[0])
    NonOscillation = len(data) - Oscillation
    return NonOscillation / Oscillation

def Gini_index(v):
    # print(v.max())
    # exit(0)
    bins = np.linspace(0.0, 100, 100)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return bins, yvals, gini_val

def analyze_csv_columns_by_index(directory_path, files, seeds, column_indexes, dim: int=6):
    f = open('./res.csv', 'w+')

    di_list = []
    gini_list = []
    file_names = []

    for file in files:
        cur_gini = []
        cur_DI = []
        file_names.append(file)
        for seed in seeds:
            if 'IS-dag' in file:
                cur_path = os.path.join(directory_path, file, str(seed), 'finetune-data.csv')
            else:
                cur_path = os.path.join(directory_path, file, str(seed), 'data.csv')
            data = pd.read_csv(cur_path)

            filename = file
            # find the target col
            for i, col_index in enumerate(column_indexes):
                if col_index < data.shape[1]:  # is limited
                    cur_data = data.iloc[:, col_index]

                    gini_index = Gini_index(cur_data)[-1]
                    cur_gini.append(gini_index)
                    cur_DI.append(get_data_imbalace_ratio(cur_data))
                    print('{filename}-gini: {}, data imbalance: {}'.format(Gini_index(cur_data)[-1], cur_DI[-1], filename=filename))
                else:
                    print(f"Column index {col_index} is out of range in {filename}")

        di_list.append("{:.2f}".format(np.mean(cur_DI)) + "±" + "{:.2f}".format(np.std(cur_DI, ddof=1)))
        gini_list.append("{:.2f}".format(np.mean(cur_gini)) + "±" + "{:.2f}".format(np.std(cur_gini, ddof=1)))

    f.writelines(",".join(["-"] + file_names) + '\n')
    f.writelines(",".join(["data imbalance"] + [str(_) for _ in di_list]) + '\n')
    f.writelines(",".join(["gini index"] + [str(_) for _ in gini_list]) + '\n')
    f.close()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default="../../../../output/mlp/reg_2/6param/HSECC")  # train data
parser.add_argument('--model_names', type=str, default="[HGGS-1w]")  # train data
parser.add_argument('--seeds', type=str, default="[53]")  # train data
parser.add_argument('--system_dimension', type=str, default="6")  # train data
args = parser.parse_args()

# HSECC
# dir_path = r'../../../../output/mlp/reg_2/6param/HSECC'
# sampling_model_names = ['HGGS-1w']
# seeds=[53]

# For brusselator: dim=2 (system coefficients)
# For the other three biological system: dim=6 (system coefficients)
# dimension = 6

dir_path = args.dir_path
model_names = eval(args.model_names)
seeds = eval(args.seeds)
system_dimension = int(args.system_dimension)

analyze_csv_columns_by_index(dir_path, model_names, seeds, [system_dimension], dim=system_dimension)
