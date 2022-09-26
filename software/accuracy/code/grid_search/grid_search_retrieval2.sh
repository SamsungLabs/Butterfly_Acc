
# Baseline: python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 64 --num_layers 2 --dropout_prob 0.1
# Explore transformer dim
python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 1.0 --transformer_dim 512 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 1.0 --transformer_dim 1024 --num_layers 2 --dropout_prob 0.1

# Explore transformer dim
python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 256 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 512 --num_layers 2 --dropout_prob 0.1
python3 run_tasks.py --model fft --task retrieval  --is_butterfly --hidden_dim_ratio 2.0 --transformer_dim 1024 --num_layers 2 --dropout_prob 0.1
