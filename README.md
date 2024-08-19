# HGGS: Hierarchical Gradient-Based Genetic Sampling

## 1. Enviroments

### Python Version

> python version=3.11.7

```bash
conda create -n HGGS python=3.11.7
```

### Requirements

```bash
conda activate HGGS
pip install -r requirements.txt
```

## 2. Structure of the Repository

### root dirctory

> **train root dirctory**: 
>
> src/yu/tasks/OnlineSamplingTools
>
> > **Baseline_Training_code:** folder contains the baseline scripts
> >
> > **Config:** training config details
> >
> > **Online_Sampling:** folder contains the HGGS scripts for training
> >
> > (Gradient-based [Ada_Gradient_init.py] and Multigrid Genetic Sampling[Genetic_Sampling.py])
> >
> > **Sampling_Algorithm:** folder contains the HGGS scripts for sampling

> **utils dictory:**
>
> src/yu/tasks/const, src/yu/tasks/core, src/yu/tasks/exception, src/yu/tasks/nn, src/yu/tasks/tools, 

> **ODEs dictory:**
>
> Brusselator system: `src/yu/tasks/model_brusselator.py `
>
> Cell Cycle system: `src/yu/tasks/model_HSECC.py`, 
>
> MPF system: `src/yu/tasks/model_MPF_2_Var.py`, 
>
> Activator Inhibitor system: `src/yu/tasks/model_PS2_01.py`, 
>
> and the **general model** `src/yu/tasks/ode_model.py`.
>
> The **oscillatory frequency is calculated** in `src/yu/tasks/pde_models.py`.

> **data root dirctory**: data/
>
> > 2param: A toy example on brusselator system (2 system coefficients)
> >
> > 6param: Three biological system, Cell Cycle (*HSECC*),  Mitotic Promoting Factor (*MPF_2_Var*), Activator Inhibitor (*PS2_01*)

> **train/sampling config dirctory**: resource/
>
> > **Training config:** OnlineSamplingTools/train.env
> >
> > **sample config (LHS):** sample/multi_generate_data.env [For multithreaded LHS]

> **pretrained model dirctory**: output/mlp/reg_2/`xparam`/`system_name`/`model_name`/`seed`/best_network.pth

## 3. Dataset

**In fact, we have already provided the training and test datasets (located in the `data/` directory). If you wish to skip this step and proceed with training the model, please refer to section 4.**

### Config

An Example: Cell Cycle System

> LHS produce data: 

LHS_code/HSECC.env

### Data Produce

**LHS code**

```bash
1. modify resource/sample/multi_generate_data.env
2. cd src/yu/tasks/sample/
3. python multi_generate_data.py
```

### Biological System

#### Brusselator System

**Cite**

> [Time, Structure, and Fluctuations | Science](https://www.science.org/doi/abs/10.1126/science.201.4358.777)
>
> In the code, we call it *brusselator*

#### Cell Cycle System

**Cite**

> [Hybrid modeling and simulation of stochastic effects on progression through the eukaryotic cell cycle - PMC (nih.gov)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3272065/)
>
> In the code, we call it *HSECC*

#### Mitotic Promoting Factor (MPF) System

**Cite**

> [Modeling the Cell Division Cycle: M-phase Trigger, Oscillations, and Size Control - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022519383711793)
>
> In the code, we call it *MPF_2_Var*

#### Activator Inhibitor System

**Cite**

> [Mathematical Biology: I. An Introduction | SpringerLink](https://link.springer.com/book/10.1007/b98868)
>
> In the code, we call it *PS2_01*

## 4. Training Model

4.1 **Cloning this repo**

```python

```

4.2 **Getting Started**

> **First:** 
>
> Create a new virtual enviroment for HGGS. 
>
> [skip into section 1 Enviroment]

> **Second:**
>
> Navigate to the directory for model training/testing.

```bash
cd src/yu/tasks/OnlineSamplingTools
```

4.3 **Training**

```bash
# HGGS-1w
python multi_seed_train.py [0,1,2,3,4,5] [6,7,8] "HGGS-1w" \
--data_path /HSECC/train/GG-Sampling/ --model_path "/HSECC/" \
--seeds "[53]" --iter_count 20 --Algorithm_type "[\"Gene_T3_Thread\", [6.0, 4.0], [\"A\", \"D\"]]" \
--finetune_epoch 0 --finetune_count 0 --tau -1 \
--epoch_n 3000 --warm_up_epoch 200 --last_epoch_n 2000 --base_lr 0.0025 --max_lr -1 \
--dataset_type boundary-Only --uniform_sampling_ratio 0.30 --boundary_sampling_ratio 0.20 --boundary_KNN 5 \
--train_strategy HGGS --ode_model_name HSECC --lr_alpha 0.2 \
--test_paths "../../../../data/6param/HSECC/val/val_5k.csv" \
--xs_param "[1.53,0.04,1.35,0.02,1.35,0.1,0.00741]" --param_selected "[0,1,2,3,4,5]" --xs_lb_ub "[0, 10]" \
--nn_layers "[128, 256, 128]" --nn_norm "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]" \
--total_training_samples 10000 --dropout 0.15 --batch_size 40960 \
--Gaussian_Mixture True --Ada_Gradient_Settings "[\"None\", 5]"
```

This code will save the model in: : **output/mlp/reg_2/`6param`/`HSECC`/`HGGS-1w`/`53`**

The remaining training scripts are located in `src/yu/tasks/train_code`

4.4 **Testing**

> testing for RMSE results

```bash
python test.py --model_name "HGGS-1w" --model_mode "HSECC/" --ode_model_name "HSECC" \
--xs_param "[1.53, 0.04, 1.35, 0.02, 1.35, 0.1, 0.00741]" --xs_param_selected "[0, 1, 2, 3, 4, 5]" \
--xs_selected "[0, 1, 2, 3, 4, 5]" --ys_selected "[6, 7, 8]"
```

You should provide the pretrained model in the path: **output/mlp/reg_2/`6param`/`HSECC`/`HGGS-1w`/`53`/best_network.pth**

The testing scripts are located in `src/yu/tasks/test_code`.

> testing for Imbalance Ratio and Gini Index results

```bash
python diversity.py
```

You can modify the `diversity.py` code between lines 68 and 74 to test other sampling models.

## 5. Pretrained Model

5.1 download `output.zip`

> All pretrained models are provied at https://drive.google.com/file/d/16bXqwkUnSJ9p--B76aoMqujarpQRsHyM/view?usp=sharing

5.2 unzip

```bash
unzip output.zip
```

> **pretrained model dirctory**: output/mlp/reg_2/`x_param`/`system_name`/`model_name`/`seed`/best_network.pth

## ~~6 **Citing Us**~~

If our work is helpful to you, please consider citing it.

```

```

## ~~7 **Acknowledgement**~~

