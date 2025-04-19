#!/usr/bin/env python3
"""Script for generating ALL experiments into experiments.txt"""
import os

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
for lr, scheduler in [(l, s) for l in lrs for s in schedulers]:
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
for lr, scheduler in [(l, s) for l in lrs for s in schedulers]:
    exp_name = f"crnn_lr_{lr}_scheduler_{scheduler}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --lr={lr} --lr_scheduler={scheduler}"
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

# Smoothers
output_dir = "smoothers"
exp_name = "none"
base_call = get_base_call(output_dir, exp_name=exp_name)
call = f"{base_call}"
print_to_file(call)
exp_name = "hmm"
base_call = get_base_call(output_dir, exp_name=exp_name)
call = (
    f"{base_call} --hmm_smoothing"
)
print_to_file(call)
exp_name = "crf"
base_call = get_base_call(output_dir, exp_name=exp_name)
call = f"{base_call} --crf "
print_to_file(call)


# Hmm alphas
output_dir = "hmm_alphas"
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for alpha in alphas:
    exp_name = f"alpha_{alpha}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --hmm_smoothing --hmm_alpha={alpha}"
    print_to_file(call)


# Spectrograms
output_dir = "spectrograms"
spectrograms = ["cqt", "mel", "linear", "chroma"]
for spectrogram in spectrograms:
    exp_name = f"spectrogram_{spectrogram}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --spectrogram_type={spectrogram}"
    print_to_file(call)

# Hop lengths
output_dir = "hop_lengths"
hop_lengths = [512, 1024, 2048, 4096, 8192, 16384]
for hop_length in hop_lengths:
    exp_name = f"hop_length_{hop_length}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --hop_length={hop_length}"
    print_to_file(call)

# Weighted alpha search
output_dir = "weight_alpha_search"
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for alpha in alphas:
    exp_name = f"alpha_{alpha}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --weight_alpha={alpha} --hmm_smoothing"
    print_to_file(call)



# Structured Loss
output_dir = "structured_loss"
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for alpha in alphas:
    exp_name = f"alpha_{alpha}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --structured_loss --weight_loss --structured_loss_alpha={alpha} --hmm_smoothing"
    print_to_file(call)


# Pitch Shifts
output_dir = "pitch_shifts"
probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for p in probabilities:
    exp_name = f"audio_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --audio_pitch_shift --aug_shift_prob={p} --hmm_smoothing --structured_loss"
    print_to_file(call)
for p in probabilities:
    exp_name = f"cqt_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --cqt_pitch_shift --aug_shift_prob={p} --hmm_smoothing --structured_loss"
    print_to_file(call)
for p in probabilities:
    exp_name = f"audio_cqt_{p}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --audio_pitch_shift --cqt_pitch_shift --aug_shift_prob={p} --hmm_smoothing --structured_loss"
    print_to_file(call)



# Generative features
output_dir = "generative_features"
# model_names = ["large", "small", "large-lerp", "melody", "chord"]
model_names = ['large']
reductions = ["concat", "avg", "codebook_0", "codebook_1", "codebook_2", "codebook_3"]
# reductions = ["codebook_3"]
for model_name, reduction in [(m, r) for m in model_names for r in reductions]:
    exp_name = f"model_{model_name}_reduction_{reduction}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --use_generative_features --gen_model_name={model_name} --gen_reduction={reduction} --hmm_smoothing --structured_loss --no_cqt --batch_size=16 --eval_batch_size=4"
    print_to_file(call)

"""
# Gen down dimension
output_dir = "gen_down_dim"
model_name = "large"
dims = [1024, 512, 256, 128, 64]
for dim in dims:
    exp_name = f"model_{model_name}_dim_{dim}"
    base_call = get_base_call(output_dir, exp_name=exp_name)
    call = f"{base_call} --weight_loss --use_generative_features --gen_model_name={model_name} --gen_down_dim={dim} --hmm_smoothing --structured_loss --gen_reduction=codebook_1 --no_cqt --batch_size=16 --eval_batch_size=4"
    print_to_file(call)

# Gen feature comparison
# output_dir = "gen_feature_comparison"
# model_name = "large"
# exp_name = "gen_only"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --use_generative_features --gen_model_name={model_name} --hmm_smoothing --structured_loss --no_cqt --gen_reduction=codebook_1 --batch_size=16 --eval_batch_size=4"
# print_to_file(call)
# exp_name = "cqt_only"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --hmm_smoothing --structured_loss"
# print_to_file(call)
# exp_name = "gen_and_cqt"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --use_generative_features --gen_model_name={model_name} --hmm_smoothing --structured_loss --gen_reduction=codebook_1 --batch_size=16 --eval_batch_size=4"
# print_to_file(call)

# Beatwise sampling
# output_dir = "beatwise_sampling"
# exp_name = "none"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --cqt_pitch_shift --structured_loss"
# print_to_file(call)
# exp_name = "feed_transitions"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --cqt_pitch_shift --input_transitions --structured_loss"
# print_to_file(call)

# beat_intervals = [0.25, 0.5]
# for beat_interval in beat_intervals:
#     exp_name = f"beat_interval_{beat_interval}"
#     base_call = get_base_call(output_dir, exp_name=exp_name)
#     call = f"{base_call} --weight_loss --cqt_pitch_shift --beat_wise_resample --beat_resample_interval={beat_interval} --structured_loss"
#     print_to_file(call)

# exp_name = "perfect_beats_train_only"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --cqt_pitch_shift --beat_wise_resample --perfect_beat_resample --structured_loss"
# print_to_file(call)

# exp_name = "perfect_beats_train_and_test"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --weight_loss --cqt_pitch_shift --beat_wise_resample --perfect_beat_resample --perfect_beat_resample_eval --structured_loss"
# print_to_file(call)

# Synthetic data
# output_dir = "synthetic_data"
# exp_name = "real_only"
# synthetic_input_dir = f"{EDDIE}/data/synthetic"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --cqt_pitch_shift --hmm_smoothing --structured_loss --test_on_synthetic --synthetic_input_dir={synthetic_input_dir}"
# print_to_file(call)
# output_dir = "synthetic_data"
# exp_name = "real_only_weighted"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --cqt_pitch_shift --weight_loss --hmm_smoothing --structured_loss --test_on_synthetic --synthetic_input_dir={synthetic_input_dir}"
# print_to_file(call)
# exp_name = "synthetic_and_real"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --cqt_pitch_shift --hmm_smoothing --structured_loss --test_on_synthetic --synthetic_input_dir={synthetic_input_dir} --use_synthetic"
# print_to_file(call)
# exp_name = "synthetic_only"
# base_call = get_base_call(output_dir, exp_name=exp_name)
# call = f"{base_call} --cqt_pitch_shift --hmm_smoothing --structured_loss --test_on_synthetic --synthetic_only --synthetic_input_dir={synthetic_input_dir} --use_synthetic"
# print_to_file(call)

# Final experiments
# output_dir = "final_experiments"



# Print number of experiments in the file
output_file.close()
with open("./scripts/experiments.txt", "r") as f:
    lines = f.readlines()
    print(f"Number of experiments: {len(lines)}")
    print("Experiments saved to ./scripts/experiments.txt")
