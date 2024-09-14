# HSECC IS
python multi_seed_train.py "IS" --model_path "/HSECC/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 200 --last_epoch_n 2000 --base_lr 0.0025 \
--ode_model_name HSECC --total_training_samples 10000 --batch_size 40960

# HSECC IS-dag
python multi_seed_train.py "IS-dag" --model_path "/HSECC/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 200 --last_epoch_n 2000 --base_lr 0.0025 \
--ode_model_name HSECC --total_training_samples 10000 --batch_size 40960
