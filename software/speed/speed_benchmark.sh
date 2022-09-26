# export CUDA_VISIBLE_DEVICES=0 to specify the gpu you. Pls specify just one gpu as we only test single gpu performance
python inference_speed.py --model_name bfly_fnet --sequence_length 128 128 128 128 128 128 128 128 128 128 --batch_size 1 --model_version base
python inference_speed.py --model_name bfly_fnet --sequence_length 256 256 256 256 256 256 256 256 256 256 --batch_size 1 --model_version base
python inference_speed.py --model_name bfly_fnet --sequence_length 512 512 512 512 512 512 512 512 512 512 --batch_size 1 --model_version base
python inference_speed.py --model_name bfly_fnet --sequence_length 768 768 768 768 768 768 768 768 768 768 --batch_size 1 --model_version base
python inference_speed.py --model_name bfly_fnet --sequence_length 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 --batch_size 1 --model_version base
python inference_speed.py --model_name bfly_fnet --sequence_length 128 128 128 128 128 128 128 128 128 128 --batch_size 1 --model_version large
python inference_speed.py --model_name bfly_fnet --sequence_length 256 256 256 256 256 256 256 256 256 256 --batch_size 1 --model_version large
python inference_speed.py --model_name bfly_fnet --sequence_length 512 512 512 512 512 512 512 512 512 512 --batch_size 1 --model_version large
python inference_speed.py --model_name bfly_fnet --sequence_length 768 768 768 768 768 768 768 768 768 768 --batch_size 1 --model_version large
python inference_speed.py --model_name bfly_fnet --sequence_length 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 --batch_size 1 --model_version large
