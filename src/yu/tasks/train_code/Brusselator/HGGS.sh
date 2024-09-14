# brusselator HGGS-1w
python multi_seed_train.py "HGGS-1w" --model_path "/brusselator/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 300 --last_epoch_n 3000 --base_lr 0.002 \
--ode_model_name brusselator --total_training_samples 10000 --batch_size 40960
