# HGGS

## 1. Enviroments

python version=3.11

## 2. Requirements

```python
pip install -r requirements.txt
```

## 3. Dataset

**In fact, we have already provided the training and test datasets (located in the `data/` directory). If you wish to skip this step and proceed with training the model, please refer to section 4.**

### Config

An Example: Cell Cycle System

> LHS produce data: 

LHS_code/HSECC.env

### Data Produce

**LHS code**

```python
1. modify resource/sample/multi_generate_data.env
2. cd src/yu/tasks/sample/
3. python multi_generate_data.py
```

### Biological System

#### Brusselator System

**Cite**

> [Time, Structure, and Fluctuations | Science](https://www.science.org/doi/abs/10.1126/science.201.4358.777)

#### Cell Cycle System

**Cite**

> [Hybrid modeling and simulation of stochastic effects on progression through the eukaryotic cell cycle - PMC (nih.gov)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3272065/)
>
> In the code, we call it *HSECC*

#### MPF System

**Cite**

> [Modeling the Cell Division Cycle: M-phase Trigger, Oscillations, and Size Control - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022519383711793)

#### Activator Inhibitor System

**Cite**

> 

## 4. Training Model

4.1 **Cloning this repo**

```python

```

4.2 **Getting Started** 

cd src/yu/tasks/OnlineSamplingTools

> training

```python
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

> testing

```python
python test.py --model_name "HGGS-1w" --model_mode "HSECC/" --ode_model_name "HSECC" \
--xs_param "[1.53, 0.04, 1.35, 0.02, 1.35, 0.1, 0.00741]" --xs_param_selected "[0, 1, 2, 3, 4, 5]" \
--xs_selected "[0, 1, 2, 3, 4, 5]" --ys_selected "[6, 7, 8]"
```

The remaining training scripts are located in `src/yu/tasks/train_code`, and the testing scripts are also in `src/yu/tasks/train_code`.

> diveristy output

```python
python diversity
```

4.3 **Citing Us**

If our work is helpful to you, please consider citing it.

```

```

4.4 **Acknowledgement**

The computation is supported by College of Computer Science and Engineering, Northeastern University. 

