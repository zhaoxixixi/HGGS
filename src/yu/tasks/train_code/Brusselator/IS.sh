# brusselator IS
python multi_seed_train.py [0,1] [2,3] "IS" \
--data_path /brusselator/train/Baseline/IS --model_path "/brusselator/" \
--seeds "[53]" --iter_count 20 --Algorithm_type "[\"Gene_T3_Thread\", [6.0, 4.0], [\"A\", \"D\"]]" \
--finetune_epoch 3000 --finetune_count 20 --tau -1 \
--epoch_n 3000 --warm_up_epoch 300 --last_epoch_n 3000 --base_lr 0.002 --max_lr -1 \
--dataset_type default --uniform_sampling_ratio 0.30 --boundary_sampling_ratio 0.20 --boundary_KNN 5 \
--train_strategy IS --ode_model_name brusselator --lr_alpha 0.2 \
--test_paths "../../../../data/2param/brusselator/val/val_5k.csv" \
--xs_param "[1, 3]" --param_selected "[0,1]" --xs_lb_ub "[0, 5]" \
--nn_layers "[128, 256, 128]" --nn_norm "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]" \
--total_training_samples 10000 --dropout 0.0 --batch_size 40960 \
--Gaussian_Mixture True --Ada_Gradient_Settings "[\"None\", 5]"

# brusselator IS-dag
python multi_seed_train.py [0,1] [2,3] "IS-dag" \
--data_path /brusselator/train/Baseline/IS --model_path "/brusselator/" \
--seeds "[53]" --iter_count 20 --Algorithm_type "[\"Gene_T3_Thread\", [6.0, 4.0], [\"A\", \"D\"]]" \
--finetune_epoch 3000 --finetune_count 20 --tau -1 \
--epoch_n 3000 --warm_up_epoch 300 --last_epoch_n 3000 --base_lr 0.002 --max_lr -1 \
--dataset_type default --uniform_sampling_ratio 0.30 --boundary_sampling_ratio 0.20 --boundary_KNN 5 \
--train_strategy IS-dag --ode_model_name brusselator --lr_alpha 0.2 \
--test_paths "../../../../data/2param/brusselator/val/val_5k.csv" \
--xs_param "[1, 3]" --param_selected "[0,1]" --xs_lb_ub "[0, 5]" \
--nn_layers "[128, 256, 128]" --nn_norm "[\"BatchNorm1d\",\"BatchNorm1d\",\"BatchNorm1d\"]" \
--total_training_samples 10000 --dropout 0.0 --batch_size 40960 \
--Gaussian_Mixture True --Ada_Gradient_Settings "[\"None\", 5]"