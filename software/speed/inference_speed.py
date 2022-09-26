# Third Party
import argparse
from genericpath import exists 
from transformers import AutoTokenizer, AutoModelForPreTraining
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, EvalPrediction
from transformers import Trainer
import os
from datetime import datetime
import time
import numpy as np
from transformers import AutoConfig, AutoModel
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# Internal Library
# from src.configs.bflylr_fnet_config import BflyLR_FNetConfig
# from src.models.bflylr_fnet_model import BflyLR_FNetForSequenceClassification

from transformers import BertForPreTraining


def compute_metrics(is_regression, metric, p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result

def train(args):

    print (args.sequence_length)
    print (type(args.sequence_length))
    ################# Initialize Model and Tokenizer ###############
    if args.model_version == "large":
        num_hidden_layers = 24
        intermediate_size = 4096
        hidden_size = 1024
        num_attention_heads = 16
    else:
        num_hidden_layers = 12
        intermediate_size = 3072
        hidden_size = 768
        num_attention_heads = 12
    # Benchmarking FNet
    if (args.model_name == "fnet"):
        from src.configs.fnet_config import Vanilla_FNetConfig
        from src.models.fnet_model import FNetForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        config = Vanilla_FNetConfig(num_attention_layers=args.num_attention_layers, attention_layout=args.attention_layout, 
                                    num_labels=args.num_labels,max_position_embeddings=args.sequence_length[-1], 
                                    tpu_short_seq_length=args.sequence_length[-1], num_hidden_layers=num_hidden_layers, 
                                    intermediate_size=intermediate_size, hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        model = FNetForSequenceClassification(config)
        print ("Vanilla Fnet")
        print (model)
        AutoConfig.register("vanilla_fnet", Vanilla_FNetConfig)
        AutoModel.register(Vanilla_FNetConfig, FNetForSequenceClassification)
        arg = PyTorchBenchmarkArguments(models=["vanilla_fnet"], batch_sizes=[args.batch_size], sequence_lengths=args.sequence_length, 
                                        memory=False, multi_process=False, fp16=args.is_fp16) 
        benchmark = PyTorchBenchmark(arg, configs=[config])
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                results = benchmark.run()
    # Benchmarking Bfly_FNet
    elif (args.model_name == "bfly_fnet"):
        from src.configs.bfly_fnet_config import Bfly_FNetConfig
        from src.models.bfly_fnet_model import Bfly_FNetForSequenceClassification
        bfly_config = Bfly_FNetConfig(num_attention_layers=args.num_attention_layers, attention_layout=args.attention_layout, 
                                    num_labels=args.num_labels, max_position_embeddings=args.sequence_length[-1], 
                                    tpu_short_seq_length=args.sequence_length[-1], num_hidden_layers=num_hidden_layers, 
                                    intermediate_size=intermediate_size, hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        bfly_model = Bfly_FNetForSequenceClassification(bfly_config)
        print ("Butterfly Fnet")
        print (bfly_model)
        AutoConfig.register("bfly_fnet", Bfly_FNetConfig)
        AutoModel.register(Bfly_FNetConfig, Bfly_FNetForSequenceClassification)
        arg = PyTorchBenchmarkArguments(models=["bfly_fnet"], batch_sizes=[args.batch_size], sequence_lengths=args.sequence_length, 
                                        memory=False, multi_process=False, fp16=args.is_fp16)
        benchmark = PyTorchBenchmark(arg, configs=[bfly_config])
        if torch.cuda.is_available(): 
            print ("Profilling CUDA performance")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    results = benchmark.run()
        else:
            print ("Profilling CPU performance")
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    results = benchmark.run()
    # Benchmarking Bert
    elif (args.model_name == "bert"):
        from src.configs.bert_config import Profile_BertConfig
        from src.models.bert_model import BertForSequenceClassification
        bert_config = Profile_BertConfig(num_labels=args.num_labels, max_position_embeddings=args.sequence_length[-1], 
                                    num_hidden_layers=num_hidden_layers, intermediate_size=intermediate_size, hidden_size=hidden_size, 
                                    num_attention_heads=num_attention_heads)
        bert_model = BertForSequenceClassification(bert_config)
        print ("Vanilla Bert")
        print (bert_model)
        AutoConfig.register("bert_vanilla", Profile_BertConfig)
        AutoModel.register(Profile_BertConfig, BertForSequenceClassification)
        arg = PyTorchBenchmarkArguments(models=["bert_vanilla"], batch_sizes=[args.batch_size], sequence_lengths=args.sequence_length, 
                                        memory=False, multi_process= False, fp16=args.is_fp16)
        benchmark = PyTorchBenchmark(arg, configs=[bert_config])
        if torch.cuda.is_available(): 
            print ("Profilling CUDA performance")
            
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    results = benchmark.run()
            # Keep this for later use, old version
            # from transformers import BertTokenizer
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', pad_to_max_length=True)
            # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt", max_length = args.sequence_length, padding='max_length')
            # bert_model.cuda()
            # inputs.to(device='cuda')
            # results = None
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            #         bert_model(**inputs)
        else: 
            print ("Profilling CPU performance")
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', pad_to_max_length=True)
            inputs = tokenizer("Hello, my dog is cute", return_tensors="pt", max_length = args.sequence_length[-1], padding='max_length')
            results = None
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    bert_model(**inputs)

    else:
        raise NotImplementedError()

    print("===================Running ", args.model_name, "=====================")
    if results is not None: print (results)
    print("===================Performance Profiling=====================")
    prof.export_chrome_trace("profile_cuda.json")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)) # cuda time is included in cpu total time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_labels", default=10, type=int, help="The number of labels used for classification")
    parser.add_argument("--num_attention_layers", default=2, type=int, help="The number of attention layers")
    parser.add_argument("--attention_layout", default="Bottom", type=str, help="The position of attention layer, One of Bottom, Top, Middle")
    parser.add_argument("--model_version", default="large", type=str, help="The version of models, base or large")
    parser.add_argument("--dataset_name", default="glue-sst2", type=str, help="The names follow the format: benchmark-subdataset")
    # Training setting according to here: https://github.com/google-research/google-research/blob/master/f_net/configs/classification.py
    parser.add_argument("--seed", default=666, type=int, help="Training random seed")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Training parameter, learning rate")
    parser.add_argument("--per_device_eval_batch_size", default=200, type=int, help="Training parameter, evaluation batch")
    parser.add_argument("--sequence_length", default=512, type=int, help="The sequence lengths", nargs="+")
    # parser.add_argument("--batch_size", default=1, type=int, help="The batch sizes", nargs="+")
    # parser.add_argument("--sequence_length", default=512, type=int, help="The sequence lengths")
    parser.add_argument("--batch_size", default=1, type=int, help="The batch sizes")
    parser.add_argument("--model_name", default="bfly_fnet", type=str, help="Support model name: bfly_fnet, fnet, bert")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int, help="Training parameter, train batch")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Training parameter, number of training epoch")
    parser.add_argument('--is_fp16', action='store_true')
    args = parser.parse_args()
    train(args)
