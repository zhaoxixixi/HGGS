# HSECC US-P
python multi_seed_train.py "US-P-5k" --model_path "/HSECC/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 200 --last_epoch_n 2000 --base_lr 0.0025 \
--ode_model_name HSECC --total_training_samples 10000 --batch_size 40960
