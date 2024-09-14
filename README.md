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
> > The **Baseline_Training_code** folder contains the baseline scripts
> >
> > **Config:** training config details
> >
> > The **Online_Sampling** folder contains the HGGS scripts for training
> >
> > (Gradient-based [Ada_Gradient_init.py] and Multigrid Genetic Sampling[Genetic_Sampling.py])
> >
> > The **Sampling_Algorithm** folder contains HGGS scripts for sampling

> **utils directory:**
>
> src/yu/tasks/const, src/yu/tasks/core, src/yu/tasks/exception, src/yu/tasks/nn, src/yu/tasks/tools.

> **ODEs directory:**
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
> > 2param: The Brusselator system (2 system coefficients)
> >
> > 6param: Data for 3 biological systems: Cell Cycle (*HSECC*),  Mitotic Promoting Factor (*MPF_2_Var*), and Activator Inhibitor (*PS2_01*)

> **train/sampling config dirctory**: resource/
>
> > **Training config:** OnlineSamplingTools/train.env and src/yu/tasks/BioSysConfig/BioSysConfig.py
> >
> > **sample config (LHS):** sample/multi_generate_data.env [For multithreaded LHS]

> **pretrained model dirctory**: output/mlp/reg_2/`xparam`/`system_name`/`model_name`/`seed`/best_network.pth

## 3. Dataset

**We have provided the training and test datasets (located in the `data/` directory). If you wish to skip this step and proceed with training the model, please refer to section 4.**

### Config

An Example: Cell Cycle System

> LHS produce data: 

LHS_code/HSECC.env

### Produce Data

**LHS code**

```bash
1. modify resource/sample/multi_generate_data.env
2. cd src/yu/tasks/sample/
3. python multi_generate_data.py
```

### Biological System

#### Brusselator System

**Citation**

> [Time, Structure, and Fluctuations | Science](https://www.science.org/doi/abs/10.1126/science.201.4358.777)
>
> In the code, we call it *brusselator*

#### Cell Cycle System

**Citation**

> [Hybrid modeling and simulation of stochastic effects on progression through the eukaryotic cell cycle - PMC (nih.gov)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3272065/)
>
> Called *HSECC* in code

#### Mitotic Promoting Factor (MPF) System

**Citation**

> [Modeling the Cell Division Cycle: M-phase Trigger, Oscillations, and Size Control - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022519383711793)
>
> Called *MPF_2_Var* in code

#### Activator Inhibitor System

**Citation**

> [Mathematical Biology: I. An Introduction | SpringerLink](https://link.springer.com/book/10.1007/b98868)
>
> Called *PS2_01* in code

## 4. Training Model

4.1 **Cloning this repo**

```python

```

4.2 **Getting Started**

>
> Create a new virtual enviroment for HGGS. 
>
> [See section 1: Enviroments]

>
> Navigate to the directory for model training/testing.

```bash
cd src/yu/tasks/OnlineSamplingTools
```

4.3 **Training**

```bash
# HGGS-1w
python multi_seed_train.py "HGGS-1w" --model_path "/HSECC/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 200 --last_epoch_n 2000 --base_lr 0.0025 \
--ode_model_name HSECC --total_training_samples 10000 --batch_size 40960
```

This code will save the model in this path from the root code directory**output/mlp/reg_2/`6param`/`HSECC`/`HGGS-1w`/`53`**

The remaining training scripts are located in (from the root) `src/yu/tasks/train_code`

4.4 **Testing**

> testing for RMSE results

```bash
python test.py --train_set "HGGS-1w" --model_path "HSECC/" --ode_model_name "HSECC" \
--seeds "[53]"
```

You should provide the pretrained model here (relative to the root code folder):**output/mlp/reg_2/`6param`/`HSECC`/`HGGS-1w`/`53`/best_network.pth**

The testing scripts are located in (relative to the root code folder) `src/yu/tasks/test_code`.

> testing for Imbalance Ratio and Gini Index results

```bash
python diversity.py --dir_path "../../../../output/mlp/reg_2/6param/HSECC" --model_names "HGGS-1w" \
--seeds "[53]" --system_dimension 6
```

## 5. Pretrained Model

5.1 download `output.zip`

> All pretrained models are provided at https://drive.google.com/file/d/16bXqwkUnSJ9p--B76aoMqujarpQRsHyM/view?usp=sharing

5.2 unzip

```bash
# move output.zip to root folder 
unzip output.zip
```

> **pretrained model directory**: output/mlp/reg_2/`x_param`/`system_name`/`model_name`/`seed`/best_network.pth
