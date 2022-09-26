from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import lra_config

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--task", type = str, help = "task", dest = "task", required = True)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--is_butterfly", help = "if enable butterfly", dest = "is_butterfly", action='store_true')
parser.add_argument("--fabnet_att_layer", type = int, help = "specify the number of attention layer used in fabnet", dest = "fabnet_att_layer", default = -1)

parser.add_argument("--hidden_dim_ratio", type = float, help = "hidden_dim = transformer_dim*hidden_dim_ratio, 0.0 keep defualt", default = 0.0)
parser.add_argument("--transformer_dim", type = int, help = "dimision of transformer, 0 keep defualt", default = 0)
parser.add_argument("--num_layers", type = int, help = "num of layers, 0 keep defualt", default = 0)
parser.add_argument("--dropout_prob", type = float, help = "dropout rate", default = 0.0)
parser.add_argument("--batch_size", type = int, help = "batch_size", default = 0)

parser.add_argument("--is_quant", help = "if apply quantization", dest = "is_quant", action='store_true')
parser.add_argument("--man_bit", type = int, help = "bitwidth of mantissa, 10 as defualt for half precision", default = 10)
parser.add_argument("--exp_bit", type = int, help = "bitwidth of exponent, 5 as defualt for half precision", default = 5)

args = parser.parse_args()

attn_type = args.model
task = args.task

checkpoint_dir = "../logs/"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print(lra_config.config[task]["extra_attn_config"].keys(), flush = True)

model_config = lra_config.config[task]["model"]
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

######################Tuning hyperparameters######################
if (args.transformer_dim != 0): 
    model_config["transformer_dim"] = args.transformer_dim
    model_config["embedding_dim"] = args.transformer_dim
if (args.hidden_dim_ratio != 0.0):
    model_config["transformer_hidden_dim"] = int(args.hidden_dim_ratio*model_config["transformer_dim"])
if (args.num_layers != 0): model_config["num_layers"] = args.num_layers
if (args.dropout_prob != 0.0): model_config["dropout_prob"] = args.dropout_prob
if (args.batch_size != 0): lra_config.config[task]["training"]["batch_size"] = args.batch_size
##################################################################

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["is_butterfly"] = args.is_butterfly
model_config["fabnet_att_layer"] = args.fabnet_att_layer
model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
model_config["is_quant"] = args.is_quant

training_config = lra_config.config[task]["training"]
gpu_memory_config = lra_config.config[task]["gpu_memory"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))

from qtorch import FloatingPoint
model_config["quant_num"] = FloatingPoint(exp=args.exp_bit, man=args.man_bit)

if task == "retrieval":
    model = ModelForSCDual(model_config)
else:
    model = ModelForSC(model_config)


print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

ds_iter = {
    "train":enumerate(DataLoader(LRADataset(f"../datasets/{task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "dev":enumerate(DataLoader(LRADataset(f"../datasets/{task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "test":enumerate(DataLoader(LRADataset(f"../datasets/{task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True)),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy = training_config["lr_decay"],
    total_steps = training_config["num_train_steps"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

def step(component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

init_t = time.time()

ratio = (str(args.hidden_dim_ratio)).replace(".", "_")
layers = (str(args.num_layers)).replace(".", "_")
dim = (str(args.transformer_dim)).replace(".", "_")
prob = (str(args.dropout_prob)).replace(".", "_")
log_f_path = os.path.join(checkpoint_dir, f"{task}_{attn_type}_output_ratio{ratio}_layer{layers}_dim{dim}_prob{prob}.log")
log_f = open(log_f_path, "a+")

summary = {
    component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
print(f"accumu_steps={accumu_steps}")

if args.is_quant:
    model.module.set_quant(True)
    print (f"Applying quantization with {args.man_bit}-bit mantissa and {args.exp_bit}-bit exponent")

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config["num_train_steps"]):
            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx)
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location = "cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()

if args.is_quant:
    model.module.set_quant(True)
    print (f"Applying quantization with {args.man_bit}-bit mantissa and {args.exp_bit}-bit exponent")

try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print_summary(summary["test"], False, train_step_idx)
