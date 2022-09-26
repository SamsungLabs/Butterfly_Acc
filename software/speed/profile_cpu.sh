export CUDA_VISIBLE_DEVICES=-1
python inference_speed.py --model_name bert --sequence_length 128 128 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 256 256 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 512 512 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 1024 1024 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 2048 2048 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 4096 4096 --batch_size 1 --model_version base 2>&1 | tee -a cpu_latency_breakdown_base.log
python inference_speed.py --model_name bert --sequence_length 128 128 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
python inference_speed.py --model_name bert --sequence_length 256 256 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
python inference_speed.py --model_name bert --sequence_length 512 512 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
python inference_speed.py --model_name bert --sequence_length 1024 1024 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
python inference_speed.py --model_name bert --sequence_length 2048 2048 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
python inference_speed.py --model_name bert --sequence_length 4096 4096 --batch_size 1 --model_version large 2>&1 | tee -a cpu_latency_breakdown_large.log
