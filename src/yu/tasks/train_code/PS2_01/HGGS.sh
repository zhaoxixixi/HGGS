# HGGS-1w
python multi_seed_train.py "HGGS-1w" --model_path "/PS2_01/" \
--seeds "[53]" --epoch_n 3000 --warm_up_epoch 250 --last_epoch_n 2500 --base_lr 0.002 \
--ode_model_name PS2_01 --total_training_samples 10000 --batch_size 40960
