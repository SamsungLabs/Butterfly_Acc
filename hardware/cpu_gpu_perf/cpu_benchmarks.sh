python benchmarking.py --model_name distilbert-base-uncased --sequence_lengths 128,256 --batch_sizes 1,4,8 
python benchmarking.py --model_name bert-base-uncased --sequence_lengths 128,256 --batch_sizes 1,4,8 
python benchmarking.py --model_name bert-large-uncased --sequence_lengths 128,256 --batch_sizes 1,4,8 
python benchmarking.py --model_name roberta-base --sequence_lengths 256 --batch_sizes 1,4,8 
python benchmarking.py --model_name gpt2 --sequence_lengths 512 --batch_sizes 1,4,8 
python benchmarking.py --model_name gpt2-medium --sequence_lengths 1024 --batch_sizes 1,4,8 
