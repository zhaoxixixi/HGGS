# HGGS-1w
python multi_seed_train.py [0,1,2,3,4,5] [6,7] "HGGS-1w" \
--data_path /MPF_2_Var/train/GG-Sampling/ --model_path "/MPF_2_Var/" \
--seeds "[53]" --iter_count 20 --Algorithm_type "[\"Gene_T3_Thread\", [6.0, 4.0], [\"A\", \"D\"]]" \
--finetune_epoch 0 --finetune_count 0 --tau -1 \
--epoch_n 3000 --warm_up_epoch 300 --last_epoch_n 3000 --base_lr 0.002 --max_lr -1 \
--dataset_type boundary-Only --uniform_sampling_ratio 0.30 --boundary_sampling_ratio 0.20 --boundary_KNN 5 \
--train_strategy HGGS --ode_model_name MPF_2_Var --lr_alpha 0.2 \
--test_paths "../../../../data/6param/MPF_2_Var/val/val_5k.csv" \
--xs_param "[0.01, 0.01, 10, 0.04, 100, 1.5, 0.1, 1]" --param_selected "[0,1,3,5,6,7]" \
--nn_layers "[128, 128, 128, 128]" --nn_norm "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]" \
--total_training_samples 10000 --dropout 0.12 --batch_size 409600 \
--Gaussian_Mixture True --Ada_Gradient_Settings "[\"None\", 5]"
