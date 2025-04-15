import autorootcwd
import os
import io
import warnings
import torch
import numpy as np
import librosa
from contextlib import redirect_stdout
from typing import Dict, List

# Jukebox imports
import jukebox.hparams
import jukebox.make_models
import jukebox.utils.dist_utils
from jukebox.make_models import MODELS
from jukebox.utils.dist_utils import setup_dist_from_mpi

# -------------------------------------------------------------------
# 1. Model Initialization
# -------------------------------------------------------------------

_SINGLETON = None  # Will store (model_name, num_layers, hps, vqvae, lm, device)

def init_jukebox_singleton(model="5b", num_layers=53, log=True):
    """
    Initializes the Jukebox model as a singleton to avoid reloading.
    Returns: (model_name, num_layers, hps, vqvae, lm, device).
    """
    global _SINGLETON
    if _SINGLETON is not None:
        # Already initialized; do a quick check if arguments match
        if (model, num_layers) != _SINGLETON[:2]:
            raise RuntimeError(
                "Jukebox is already initialized with different parameters!"
            )
        return _SINGLETON

    # Set up distributed environment
    silent_buffer = io.StringIO()
    with redirect_stdout(silent_buffer):
        rank, local_rank, device = setup_dist_from_mpi()

    if log:
        print(silent_buffer.getvalue())

    # Hyperparams
    hps = jukebox_features.hparams.Hyperparams()
    hps.sr = 44100  # Jukebox default
    hps.n_samples = 3 if model == "5b_lyrics" else 8
    hps.name = "samples"
    chunk_size = 16 if model == "5b_lyrics" else 32
    max_batch_size = 3 if model == "5b_lyrics" else 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]

    # Load VQVAE (encoder-decoder)
    vqvae_name, *prior_names = MODELS[model]
    vqvae_hps = jukebox_features.hparams.setup_hparams(vqvae_name, dict(sample_length=0))
    silent_buffer = io.StringIO()
    with redirect_stdout(silent_buffer):
        vqvae = jukebox_features.make_models.make_vqvae(vqvae_hps, device)
    if log:
        print(silent_buffer.getvalue())

    # Load top-level prior
    # num_layers = number of transformer blocks in the top-level prior
    top_prior_name = prior_names[-1]  # top-level prior
    overrides = dict(prior_depth=num_layers) if num_layers else {}
    top_prior_hps = jukebox_features.hparams.setup_hparams(top_prior_name, overrides)
    silent_buffer = io.StringIO()
    with redirect_stdout(silent_buffer):
        lm = jukebox_features.make_models.make_prior(top_prior_hps, vqvae, device)
    if log:
        print(silent_buffer.getvalue())

    lm.prior.only_encode = True  # We'll only use it to encode

    _SINGLETON = (model, num_layers, hps, vqvae, lm, device)
    return _SINGLETON

def get_jukebox_model(
    model_name: str = "5b",
    num_layers: int = 53,
    device: str = "cuda",
    log: bool = True
):
    """
    Returns a pretrained Jukebox VQVAE + Prior model for representation extraction.
    """
    # Initialize Jukebox
    # This sets up device, loads VQVAE, and loads top-level prior
    model, layers, hps, vqvae, lm, loaded_device = init_jukebox_singleton(
        model=model_name, num_layers=num_layers, log=log
    )

    # We can override if user sets device="cpu", but be warned it's slow
    if loaded_device.type != device:
        print(f"[Warning] Jukebox was loaded on {loaded_device}, not {device}.")

    return (hps, vqvae, lm, loaded_device)


# -------------------------------------------------------------------
# 2. Audio Loading
# -------------------------------------------------------------------

def get_wav(
    filename: str,
    dir: str = "./data/processed/",
    target_sr: int = 44100,
    mono: bool = True
):
    """
    Loads audio from dir/filename using librosa, resamples to target_sr if needed.
    Returns:
       audio (np.ndarray): shape [samples,] if mono=True,
                           or [channels, samples] if mono=False
       sr (int): sample rate
    """
    path = os.path.join(dir, filename)
    # If it's .mp3 or .wav, etc., librosa.load can handle it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(path, sr=None, mono=False)  # preserve stereo if present

    # If mono is desired, downmix
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=0, keepdims=False)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


# -------------------------------------------------------------------
# 3. Overlapping Chunking + Representation Extraction
# -------------------------------------------------------------------

def extract_song_hidden_representation_jukebox(
    filename: str,
    dir: str,
    jukebox_bundle,
    max_chunk_length: float = 10.0,
    overlap_ratio: float = 0.5,
    fp16: bool = False,
    frame_length: float = None,
):
    """
    Process an entire song in overlapping chunks using Jukebox’s VQVAE + top-level prior,
    and return some representation. This mirrors the approach in the MusicGen code.

    Args:
        filename (str): Name of audio file (e.g., "my_song.mp3").
        dir (str): Directory containing the file.
        jukebox_bundle (tuple): The (hps, vqvae, lm, device) from get_jukebox_model().
        max_chunk_length (float): The chunk length in seconds for processing.
        overlap_ratio (float): Overlap fraction between consecutive chunks [0..1].
        fp16 (bool): If True, do forward pass in float16 to save memory.
        resample_frame_length (float): If not None, desired time resolution in seconds
                                       for final representation. (We can optionally
                                       implement an interpolate if needed, akin to
                                       MusicGen’s `resample_hidden_states`.)

    Returns:
        A dictionary of results. Example: { "concat": <torch.Tensor of shape [T, D]> }
    """
    hps, vqvae, lm, device = jukebox_bundle

    # 1) Load audio
    audio_np, sr = get_wav(filename, dir=dir, target_sr=hps.sr, mono=True)
    audio_len = audio_np.shape[-1]  # samples
    total_duration = audio_len / float(sr)

    # 2) Determine chunk size in samples + hop
    chunk_samples = int(max_chunk_length * sr)
    hop_samples = int(chunk_samples * (1 - overlap_ratio))
    if hop_samples < 1:
        raise ValueError("overlap_ratio too large, resulting in 0 hop.")

    # 3) Prepare start indices for each chunk
    start_indices = []
    idx = 0
    while idx < audio_len:
        start_indices.append(idx)
        idx += hop_samples
    # Last chunk might go beyond audio_len, we handle that with padding if needed

    # 4) We'll store aggregated hidden states for the entire track
    #    We must figure out the "frames" dimension for Jukebox’s top-level prior.
    #    The top-level prior works at the top-level codes. Each code frame
    #    corresponds to a certain # of audio samples = hop_length * 128, etc.
    #    But let's do something simpler: we’ll build an array of shape [T_total, D].
    #    Then accumulate from each chunk with overlap weighting.

    # Decide on “frames per second” or “frames per chunk” from top-level prior.
    # Jukebox’s top prior often expects chunks of shape [B, T_chunk].
    # The top-level code rate is (sample_rate / hop_size), with hop_size=128 for 5B.
    # So we can define:
    top_level_hop = 128  # for 5B. If you're using a different model, verify.

    # Approx total frames
    total_frames = audio_len // top_level_hop
    # We'll keep a buffer for hidden states [total_frames, D]
    # But we won't know D until we do a forward pass. We’ll wait to create it.

    # 5) Helper to encode + forward a single chunk
    def jukebox_forward_chunk(audio_chunk_np):
        """
        Encode audio chunk with VQVAE, then run top-level prior in `only_encode` mode
        to get hidden states. Returns (activations, mask).
         - activations: shape [T_chunk, D]
        """
        # Ensure shape [1, samples]
        if audio_chunk_np.ndim == 1:
            audio_chunk_np = audio_chunk_np[None, :]
        # Convert to torch
        audio_t = torch.tensor(audio_chunk_np, device=device, dtype=torch.float32)
        # VQVAE encode returns codes at 3 levels, we need the top-level codes => index -1
        with torch.no_grad():
            codes = vqvae.encode(audio_t)[-1]  # shape [1, T_codes]
        # Now feed codes into the top prior
        # The “forward” method if `only_encode=True` will produce embeddings from the final layer
        with torch.no_grad():
            # x_cond, y_cond = None, None => or we can pass metadata if we want
            # Jukebox 5B top prior expects shape [B, T], so codes is [1, T_codes].
            if fp16:
                codes = codes.half()
            # The forward pass below typically yields shape [1, T_codes, D]
            # in `only_encode` mode
            activations = lm.prior.forward(codes, x_cond=None, y_cond=None, fp16=fp16)
            # Convert to float32 if needed
            if fp16:
                activations = activations.float()
        # Remove batch dim => [T_codes, D]
        activations = activations.squeeze(0)
        return activations

    # 6) Process chunks in a loop, accumulate results
    # We first do one chunk to figure out dimension D
    test_chunk = audio_np[:chunk_samples]
    if test_chunk.shape[0] < chunk_samples:
        test_chunk = np.pad(test_chunk, (0, chunk_samples - test_chunk.shape[0]))
    test_acts = jukebox_forward_chunk(test_chunk)
    D = test_acts.shape[-1]

    # Initialize an accumulation buffer
    # shape => [total_frames, D]
    # We also keep a weight buffer to handle overlaps
    accum = torch.zeros(total_frames, D, device=device)
    weight = torch.zeros(total_frames, 1, device=device)

    # 7) We do chunking in a simple for-loop. For large audio, you might want to
    #    do it in mini-batches, but we’ll keep it simpler for demonstration:
    for start in start_indices:
        end = min(start + chunk_samples, audio_len)
        chunk = audio_np[start:end]
        # Pad if needed
        if chunk.shape[0] < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - chunk.shape[0]))

        # Forward pass => [T_chunk, D]
        acts_chunk = jukebox_forward_chunk(chunk)
        T_chunk = acts_chunk.shape[0]

        # Figure out “global frame” start in the final buffer
        global_start = start // top_level_hop
        global_end = global_start + T_chunk
        if global_start >= total_frames:
            break
        if global_end > total_frames:
            T_valid = total_frames - global_start
            acts_chunk = acts_chunk[:T_valid, :]
            T_chunk = T_valid

        # Accumulate
        accum[global_start : global_start + T_chunk] += acts_chunk
        weight[global_start : global_start + T_chunk] += 1.0

    # 8) Divide by weight to get average in overlapped regions
    nonzero_mask = weight.squeeze(-1) > 0
    accum[nonzero_mask] = accum[nonzero_mask] / weight[nonzero_mask]

    # 9) If you’d like to resample the timeline to a different frame length in seconds,
    #    do an interpolation step. For example, if resample_frame_length=0.1, that means
    #    10 frames/sec. Jukebox top-level is typically (sr / 128) ~ 344.53 frames/sec,
    #    so you might want to downsample drastically to reduce dimension.
    #    This is optional. Here’s a minimal approach:

    if frame_length is not None:
        # frames/sec for top-level prior
        fps = sr / float(top_level_hop)
        # We have shape [total_frames, D] => interpret it as [1, T, D] for interpolation
        accum_3d = accum.unsqueeze(0).permute(0, 2, 1)  # => [1, D, T]
        input_duration_sec = total_frames / fps
        # desired number of frames
        new_T = int(round(input_duration_sec / frame_length))

        # "area" or "linear" mode
        resampled_3d = torch.nn.functional.interpolate(
            accum_3d, size=new_T, mode="area"  # area= mean pooling
        )
        # => [1, D, new_T]
        resampled_3d = resampled_3d.permute(0, 2, 1)  # => [1, new_T, D]
        final_rep = resampled_3d.squeeze(0)  # => [new_T, D]
    else:
        final_rep = accum  # => [total_frames, D]

    # 10) Return a dictionary similar to your MusicGen approach
    result = {
        "codebook_0": final_rep.cpu(),  # or keep on GPU if you want
    }
    return result