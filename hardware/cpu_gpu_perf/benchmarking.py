import os
import argparse 
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

def run_benchmark(args):
    models = args.model_name.split(",")
    batches = args.batch_sizes.split(",")
    batches = [int(batch) for batch in batches]
    sequence_lengths = args.sequence_lengths.split(",")
    sequence_lengths = [int(sequence_length) for sequence_length in sequence_lengths]

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu is None else args.gpu   

    config = PyTorchBenchmarkArguments(models=models, batch_sizes=batches, sequence_lengths=sequence_lengths)
    benchmark = PyTorchBenchmark(config)
    results = benchmark.run()
    print (results)

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="bert-base-uncased", type=str,
                        help="Specify the transformer model, can be multile seperate by comma}")
    parser.add_argument("--sequence_lengths", default="128", type=str, help="The sequence length of inputs, can be multiple seperate by comma")
    parser.add_argument("--batch_sizes", default="1", type=str, help="Batch size,  can be multiple seperate by comma")
    parser.add_argument("--gpu", default=None, type=str, help="Specify gpu")

    args = parser.parse_args()
    run_benchmark(args)
