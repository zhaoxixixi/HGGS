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
