
# Baseline: python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.1
# Explore hidden ratio
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 1.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 4.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.1

# Explore transformer dim
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 128 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 256 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 512 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 1024 --num_layers 2 --dropout_prob 0.1

# Explore number of layer
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 4 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 6 --dropout_prob 0.1
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 8 --dropout_prob 0.1

# Explore number of dropout rate
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.2
python3 run_tasks.py --model fft --task listops  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.4
