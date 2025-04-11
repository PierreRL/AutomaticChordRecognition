#!/usr/bin/env python3
"""Script for generating ALL experiments into experiments.txt"""
import os
from itertools import product

# The home dir on the node's scratch disk
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

output_file = open("./scripts/experiments.txt", "w")


def get_base_call(output_dir, exp_name=None):
    return f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={EDDIE}/experiments/{output_dir} --exp_name={exp_name}"


def print_to_file(expt):
    """Prints the experiment to a file"""
    print(expt, file=output_file)


# Logistic LR Search
output_dir = "logistic_lr_search"
lrs = [0.0001, 0.001, 0.01, 0.1]
schedulers = ["cosine", "plateau", "none"]

for lr, scheduler in product(lrs, schedulers):
    exp_name = f"lr_{lr}_scheduler_{scheduler}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --model=logistic --lr={lr} --scheduler={scheduler}"
    print_to_file(call)

# CNNs
output_dir = "cnns"
kernel_sizes = [5, 5, 9]
layers = [1, 3, 5]
channels = [1, 5, 10]
for k, l, c in zip(kernel_sizes, layers, channels):
    exp_name = f"cnn_k{k}_l{l}_c{c}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = (
        f"{base_call} "
        f"--model=cnn "
        f"--cnn_kernel_size={k} "
        f"--cnn_layers={l} "
        f"--cnn_channels={c} "
    )
    print_to_file(call)

# LR search on CRNN
output_dir = "crnn_lr_search"
lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
schedulers = ["cosine", "plateau", "none"]
for lr, scheduler in product(lrs, schedulers):
    exp_name = f"crnn_lr_{lr}_scheduler_{scheduler}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --model=crnn --lr={lr} --scheduler={scheduler}"
    print_to_file(call)

# CRNN Hparams
output_dir = "crnn_hparams"
