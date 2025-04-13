#!/usr/bin/env python3
"""Script for generating ALL experiments into experiments.txt"""
import os
from itertools import product
import numpy as np

USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")
REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

if EDDIE is None:
    EDDIE = REPO_HOME  # Compatibility with eddie and mlt clusters

DATA_HOME = f"{EDDIE}/data/processed"

output_file = open("./scripts/experiments.txt", "w")


def get_base_call(output_dir, exp_name=None):
    return f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={EDDIE}/experiments/{output_dir} --exp_name={exp_name}"


def print_to_file(expt):
    """Prints the experiment to a file"""
    print(expt, file=output_file)


"""
# Logistic LR Search
output_dir = "logistic_lr_search"
lrs = [0.0001, 0.001, 0.01, 0.1]
schedulers = ["cosine", "plateau", "none"]
for lr, scheduler in product(lrs, schedulers):
    exp_name = f"lr_{lr}_scheduler_{scheduler}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --model=logistic --lr={lr} --lr_scheduler={scheduler}"
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
    call = f"{base_call} --lr={lr} --lr_scheduler={scheduler} --hidden_size=256 --segment_length=8"
    print_to_file(call)

# CRNN Hparams random search
output_dir = "crnn_hparams"
kernel_sizes = [5, 15]
cnn_layers = [1, 5]
channels = [1, 5]
hidden_sizes = [32, 512]
segment_length = [5, 45]
gru_layers = [1, 3]
num_expts = 50
for _ in range(num_expts):
    # Choose int within range of each hyperparameter e.g. 5-9 for kernel size
    k = np.random.randint(kernel_sizes[0], kernel_sizes[1])
    l = np.random.randint(cnn_layers[0], cnn_layers[1])
    c = np.random.randint(channels[0], channels[1])
    h = np.random.randint(hidden_sizes[0], hidden_sizes[1])
    s = np.random.randint(segment_length[0], segment_length[1])
    r = np.random.randint(gru_layers[0], gru_layers[1])

    exp_name = f"crnn_k{k}_l{l}_c{c}_h{h}_s{s}_r{r}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = (
        f"{base_call} "
        f"--cnn_kernel_size={k} "
        f"--cnn_layers={l} "
        f"--cnn_channels={c} "
        f"--hidden_size={h} "
        f"--segment_length={s} "
        f"--gru_layers={r} "
    )
    print_to_file(call)

# Long SGD
exp_name = "long_sgd"
base_call = get_base_call("", exp_name=exp_name)
call = f"{base_call} --optimiser=sgd --epochs=2000"
print_to_file(call)

# Hop lengths
output_dir = "hop_lengths"
hop_lengths = [512, 1024, 2048, 4096, 8192, 16384]
for hop_length in hop_lengths:
    exp_name = f"hop_length_{hop_length}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --hop_length={hop_length}"
    print_to_file(call)

# Small vs Large vocab
output_dir = "small_vs_large_vocab"
# base_call = get_base_call(output_dir, exp_name="small") Doesn't work on cluster
# call = f"{base_call} --small_vocab=True"
# print_to_file(call)
base_call = get_base_call(output_dir, exp_name="large")
call = f"{base_call}"
print_to_file(call)

# CR2
output_dir = "cr2"
base_call = get_base_call(output_dir, exp_name="cr2_on")
call = f"{base_call} --cr2"
print_to_file(call)
base_call = get_base_call(output_dir, exp_name="cr2_off")
call = f"{base_call}"
print_to_file(call)

# Weighted alpha search
output_dir = "weight_alpha_search"
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for alpha in alphas:
    exp_name = f"alpha_{alpha}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --weight_alpha={alpha}"
    print_to_file(call)
"""

# Pitch Shifts
output_dir = "pitch_shifts"
probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for p in probabilities:
    exp_name = f"audio_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --audio_pitch_shift --aug_shift_prob={p}"
    print_to_file(call)
for p in probabilities:
    exp_name = f"cqt_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} ---weight_loss --cqt_pitch_shift --aug_shift_prob={p}"
    print_to_file(call)
for p in probabilities:
    exp_name = f"audio_cqt_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --audio_pitch_shift --cqt_pitch_shift --aug_shift_prob={p}"
    print_to_file(call)

# Structured Loss
# output_dir = "structured_loss"
# alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# for alpha in alphas:
#     exp_name = f"alpha_{alpha}"
#     base_call = get_base_call(output_dir, exp_name=exp_name)
#     call = f"{base_call} --structured_loss --weight_loss --structured_loss_alpha={alpha} --aug_shift_prob=0.5 --audio_pitch_shift --cqt_pitch_shift"
#     print_to_file(call)

# Smoother

# Print number of experiments in the file
output_file.close()
with open("./scripts/experiments.txt", "r") as f:
    lines = f.readlines()
    print(f"Number of experiments: {len(lines)}")
    print("Experiments saved to ./scripts/experiments.txt")
