"""
Contains functions for creating generative features of songs. 

Inspired by work in: https://arxiv.org/pdf/2107.05677

We adapt their method to use with a newer model, MusicGen.
"""

import autorootcwd
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes

from src.utils import get_torch_device, get_filenames

def get_musicgen_model(model_size: str, device: str = "cuda"):
    """
    Returns a pretrained MusicGen model.

    Args:
    - model_size (str): The size of the model to load. Either 'small' or 'large'.

    Returns:
    - model (MusicGen): The pretrained model.
    """
    assert model_size in ["small", "large"], "Model size must be 'small' or 'large'."
    # model = MusicGen.get_pretrained('facebook/musicgen-' + model_size, device=device)
    local_path = os.path.expanduser(f"/exports/eddie/scratch/s2147950/.cache/huggingface/musicgen-{model_size}")
    model = MusicGen.get_pretrained(local_path, device=device)
    model.lm = model.lm.float()
    model.compression_model = model.compression_model.float()
    model.lm.eval()
    model.compression_model.eval()
    return model

def get_wav(filename: str, dir = "./data/processed/", device = "cuda", target_sr = 32000):
    """
    Loads a wav file from the given directory. Resamples the audio to 32kHz if the model is not None and converts to mono.

    Args:
    - filename (str): The name of the file to load.
    - dir (str): The directory to load the file from.

    Returns:
    - wav (torch.Tensor): The audio waveform.
    - sr (int): The sample rate of the audio.
    """
    # Load the audio file.
    wav, sr = torchaudio.load(f"{dir}/audio/{filename}.mp3")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # convert to mono

    wav = wav.unsqueeze(0).to(device)  # shape becomes [1, channels=1, samples]

    # Resample to target sample rate if provided.
    # Note: MusicGen models are trained on 32kHz audio.
    if target_sr is not None and sr != target_sr:
        resampler = Resample(orig_freq=sr, new_freq=target_sr).to(device)
        wav = resampler(wav)

    return wav, target_sr

def resample_hidden_states(
    hidden_states: torch.Tensor,
    model,
    desired_frame_length: float
) -> torch.Tensor:
    """
    Resample hidden state tensor to a new temporal resolution given a desired frame length.
    
    Args:
        hidden_states (torch.Tensor): Tensor of shape [B, T, D] representing hidden states.
        model: The MusicGen model (used to obtain model.compression_model.frame_rate if input_duration is not provided).
        desired_frame_length (float): Desired time interval between frames (in seconds). 
            For example, 4096/44100 (~0.093 s) yields ~10-11 frames per second.
    
    Returns:
        torch.Tensor: Resampled hidden states of shape [B, new_T, D],
            where new_T = round(input_duration / desired_frame_length).
    """
    B, T, D = hidden_states.shape

    current_frame_rate = model.compression_model.frame_rate
    input_duration = T / current_frame_rate  # in seconds

    # Compute the new number of time steps so that each represents the desired frame length.
    new_T = int(round(input_duration / desired_frame_length))

    # Permute to [B, D, T] because F.interpolate operates on the last dimension as time.
    hidden_states_perm = hidden_states.permute(0, 2, 1)
    
    # Use linear interpolation along the time dimension.
    resampled = F.interpolate(hidden_states_perm, size=new_T, mode='linear', align_corners=False)
    
    # Permute back to [B, new_T, D]
    resampled = resampled.permute(0, 2, 1)
    
    return resampled


def extract_song_hidden_representation(
    filename: List[str],
    dir: str,
    model,
    max_chunk_length: float,
    frame_length: float = None,
    overlap_ratio: float = 0.5,
    max_batch_size: int = 16
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Process an entire song in overlapping chunks and returns a hidden state representation
    for the entire song resampled to have one vector every desired_frame_length seconds.
    
    This function uses the previously defined helper functions:
      - get_intermediate_hidden_state(model, prompt_tokens, text="a song", layer_frac=None)
      - resample_hidden_states(hidden_states, model, desired_frame_length, input_duration)
    
    Args:
        wav_file (str): Path to the wav file.
        model: A pretrained MusicGen model.
        max_chunk_length (float): Maximum chunk length in seconds (the LM can only process a limited duration at a time).
        desired_frame_length (float): Desired time interval (in seconds) between hidden state vectors in the final output.
                                      For example, 4096/44100 (~0.093 s) would yield ~10–11 frames per second.
        overlap_ratio (float, optional): Fraction of overlap between consecutive chunks (default 0.5 for 50% overlap).
        layer_indices (list, optional): List of layer indices to extract the hidden state from. 1-indexed.
        
    Returns:
        torch.Tensor: A hidden state tensor for the entire song, of shape [new_T, D],
                      where T_final ≈ (total_duration / desired_frame_length) and D is the hidden dimension.
    """
    device = get_torch_device(allow_mps=False)

    model.compression_model = model.compression_model.to(device)

    # Load the audio file.
    wav, sr = get_wav(filename, dir=dir, device=device, target_sr=model.sample_rate)

    total_samples = wav.shape[-1]
    total_duration = total_samples / sr  # in seconds

    # Get the frame rate from the compression model.
    fps = model.compression_model.frame_rate  # frames per second, e.g., 50 fps.
    # Global number of frames corresponding to the entire audio (before resampling).
    global_frames = int(round(total_duration * fps))
    
    # Get hidden state dimensionality (assume from first embedding layer).
    D = model.lm.emb[0].embedding_dim
    
    # Determine chunking parameters.
    chunk_samples = int(max_chunk_length * sr)  # chunk length in samples.
    hop_samples = int(chunk_samples * (1 - overlap_ratio))  # hop size between chunks.

    # Conditioning attributes for the LM.
    neutral_condition = ConditioningAttributes(text={'description': ''})
    tokenized = model.lm.condition_provider.tokenize([neutral_condition])
    condition_tensors = model.lm.condition_provider(tokenized)

    # Get list of start indices for chunks.
    start_indices = []
    idx = 0
    while idx < total_samples:
        start_indices.append(idx)
        idx += hop_samples
    
        # Create list of chunks; pad last chunk if needed.
    chunk_list = []
    for start in start_indices:
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, :, start:end]  # [1, channels, L]
        if chunk.shape[-1] < chunk_samples:
            pad_amount = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount))
        chunk_list.append(chunk)  # each chunk is [1, channels, chunk_samples]
    
    num_chunks = len(chunk_list)

        # Global accumulators for each codebook.
    K = model.lm.num_codebooks  # number of codebooks (e.g., 4)
    # Use LM linear layer output features as "card"
    card = model.lm.linears[0].out_features
    global_logits = torch.zeros(K, global_frames, card, device=device)
    global_weights = torch.zeros(1, global_frames, 1, device=device)
    

    # Process chunks in batches to limit memory usage.
    for i in range(0, num_chunks, max_batch_size):
        batch_chunks = torch.cat(chunk_list[i : i + max_batch_size], dim=0)  # [B, channels, chunk_samples] where B <= max_batch_size
        # Pass through compression model.
        with torch.no_grad():
            # codes shape: [B, K, S_chunk]
            codes, _ = model.compression_model.encode(batch_chunks)
        # Prepare conditions for each chunk in batch.
        B = codes.shape[0]
        conditions = [ConditioningAttributes(text={'description': 'a song'}) for _ in range(B)]
        # conditions = []
        with torch.no_grad():
            # LM compute_predictions: logits shape: [B, K, T_chunk, card]
            lm_output = model.lm.compute_predictions(codes, conditions=conditions, stage=-1, keep_only_valid_steps=True, 
            # condition_tensors=condition_tensors
            )
            batch_logits = lm_output.logits  # [B, K, T_chunk, card]

        # For each chunk in the batch, accumulate logits.
        T_chunk = batch_logits.shape[2]
        for j in range(B):
            chunk_global_start = int(round((start_indices[i + j] / sr) * fps))
            if chunk_global_start + T_chunk > global_frames:
                valid_T = global_frames - chunk_global_start
                chunk_logits = batch_logits[j, :, :valid_T, :]
            else:
                valid_T = T_chunk
                chunk_logits = batch_logits[j, :, :valid_T, :]
            for k in range(K):
                global_logits[k, chunk_global_start:chunk_global_start + valid_T, :] += chunk_logits[k, :valid_T, :]
            global_weights[0, chunk_global_start:chunk_global_start + valid_T, 0] += 1.0

    # Average overlapping regions.
    global_logits = global_logits / global_weights  # shape: [K, global_frames, card]

    resampled_per_codebook = []
    for k in range(K):
        logits_k = global_logits[k].unsqueeze(0)  # [1, global_frames, card]
        resampled = resample_hidden_states(logits_k, model, frame_length)  # [1, new_T, card]
        resampled_per_codebook.append(resampled.squeeze(0))  # [new_T, card]

   # Compute all reductions.
    # Stack across codebooks → [new_T, K, card]
    stacked = torch.stack(resampled_per_codebook, dim=1)
    
    # "avg": average across codebooks -> [new_T, card]
    avg_rep = stacked.mean(dim=1)  # [new_T, card]
    
    # "concat": concatenate along the feature dimension.
    concat_rep = torch.cat([stacked[:, k, :] for k in range(K)], dim=-1)  # [new_T, K * card]
    
    # "first": provide each codebook separately.
    single_dict = {}
    for k in range(K):
        single_dict[f"codebook_{k}"] = resampled_per_codebook[k]  # each is [new_T, card]
    
    # Return a dict containing all reduction methods.
    result = {
        "avg": avg_rep,
        "concat": concat_rep,
        "single": {k: single_dict[k].cpu().detach().numpy() for k in single_dict}
    }
    return result




"""
Legacy behaviour in extracting hidden states from layers within the model. Warning: not tested thoroughly! It does run but the representations may not function as expected.
"""

# def get_hidden_states_from_tokens(model, prompt_tokens, text="a song", layer_indices=[18]):
#     """
#     Given a MusicGen model and a token tensor (shape [B, K, S]),
#     returns the LM transformer hidden state at a specified intermediate layer.
    
#     Args:
#         model: A pretrained MusicGen model (loaded via MusicGen.get_pretrained).
#         prompt_tokens (torch.Tensor): Token tensor from the compression model of shape [B, K, S],
#             where K is the number of codebooks.
#         text (str): Text condition to use. For unconditional or generic behavior, e.g. "a song".
#         layer_indices (list): List of layer indices to extract the hidden state from. 1-indexed.
    
#     Returns:
#         torch.Tensor: The hidden state from the specified transformer layer, shape [B, S, embed_dim].
#     """
#     # Dimensions: B=batch, K=number of codebooks, S=sequence length.
#     B, K, S = prompt_tokens.shape
    
#     # Compute input embeddings by summing embeddings from each codebook.
#     input_emb = sum(model.lm.emb[k](prompt_tokens[:, k]) for k in range(K))
    
#     # Build conditioning attributes.
#     conditions = [ConditioningAttributes(text={'description': text}) for _ in range(B)]
    
#     # Process conditions
#     conditions = model.lm.cfg_dropout(conditions)
#     conditions = model.lm.att_dropout(conditions)
#     tokenized = model.lm.condition_provider.tokenize(conditions)
#     condition_tensors = model.lm.condition_provider(tokenized)
    
#     # Fuse the audio embeddings with the conditioning.
#     fused_input, cross_attention_input = model.lm.fuser(input_emb, condition_tensors)
#     # fused_input shape: [B, S, embed_dim]

#     fused_input = fused_input.float()
#     cross_attention_input = cross_attention_input.float()
    
#     # Prepare the transformer input by adding positional embeddings.
#     transformer = model.lm.transformer  # This is a 'StreamingTransformer'.
#     B, T, C = fused_input.shape
#     # For a fixed (non-streaming) chunk, we can assume offsets = 0.
#     offsets = torch.zeros(B, dtype=torch.long, device=fused_input.device)
#     if transformer.positional_embedding in ['sin', 'sin_rope']:
#         positions = torch.arange(T, device=fused_input.device).view(1, -1, 1)
#         positions = positions + offsets.view(-1, 1, 1)
#         pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=fused_input.dtype)
#         x = fused_input + transformer.positional_scale * pos_emb
#     else:
#         x = fused_input

#     x = x.float()  # Ensure the input is in float32 format.

#     # Determine which layer to extract.
#     num_layers = len(transformer.layers)
#     if layer_indices is None:
#         layer_indices = [num_layers]  # Default to the last layer.

#     layer_indices = [idx - 1 for idx in layer_indices]  # Convert to 0-indexed.
#     assert [idx for idx in layer_indices if idx < 0 or idx >= num_layers] == [], \
#         f"Layer indices must be in the range [0, {num_layers - 1}]."

#     hidden_states_dict = {}
    
#     # 7. Iterate through transformer layers until the specified index.
#     for idx, layer in enumerate(transformer.layers):
#         # We pass cross_attention_src and optionally src_mask.
#         # Here we use stage=-1 (i.e. no specific codebook mask) as in LMModel.forward.
#         x = transformer._apply_layer(
#             layer,
#             x,
#             cross_attention_src=cross_attention_input,
#             src_mask=None
#         )
#         if idx in layer_indices:
#             # Add the 1 to the index to match the original 1-indexed layer indices.
#             hidden_states_dict[idx+1] = x.clone()  # Store the hidden state for this layer.
#     # If layer_index is greater than available layers, return the final hidden state.
#     return hidden_states_dict

# def extract_song_hidden_representation(
#     filename: str,
#     dir: str,
#     model,
#     max_chunk_length: float,
#     frame_length: float,
#     overlap_ratio: float = 0.5,
#     layer_indices: List[int] = [18]
# ) -> torch.Tensor:
#     """
#     Process an entire song in overlapping chunks and returns a hidden state representation
#     for the entire song resampled to have one vector every desired_frame_length seconds.
    
#     This function uses the previously defined helper functions:
#       - get_intermediate_hidden_state(model, prompt_tokens, text="a song", layer_frac=None)
#       - resample_hidden_states(hidden_states, model, desired_frame_length, input_duration)
    
#     Args:
#         wav_file (str): Path to the wav file.
#         model: A pretrained MusicGen model.
#         max_chunk_length (float): Maximum chunk length in seconds (the LM can only process a limited duration at a time).
#         desired_frame_length (float): Desired time interval (in seconds) between hidden state vectors in the final output.
#                                       For example, 4096/44100 (~0.093 s) would yield ~10–11 frames per second.
#         overlap_ratio (float, optional): Fraction of overlap between consecutive chunks (default 0.5 for 50% overlap).
#         layer_indices (list, optional): List of layer indices to extract the hidden state from. 1-indexed.
        
#     Returns:
#         torch.Tensor: A hidden state tensor for the entire song, of shape [new_T, D],
#                       where T_final ≈ (total_duration / desired_frame_length) and D is the hidden dimension.
#     """
#     device = get_torch_device(allow_mps=False)

#     model.compression_model = model.compression_model.to(device)

#     # Load the audio file.
#     wav, sr = get_wav(filename, dir=dir, device=device, target_sr=model.sample_rate)

#     total_samples = wav.shape[-1]
#     total_duration = total_samples / sr  # in seconds

#     # Get the frame rate from the compression model.
#     fps = model.compression_model.frame_rate  # frames per second, e.g., 50 fps.
#     # Global number of frames corresponding to the entire audio (before resampling).
#     global_frames = int(round(total_duration * fps))
    
#     # Get hidden state dimensionality (assume from first embedding layer).
#     D = model.lm.emb[0].embedding_dim

#     # Preallocate accumulators for the global hidden state and a weight mask.
#     global_hidden_dict = {idx: torch.zeros(1, global_frames, D, device=device)
#                           for idx in layer_indices}
#     global_weights = torch.zeros(1, global_frames, 1, device=device)
    
#     # Determine chunking parameters.
#     chunk_samples = int(max_chunk_length * sr)  # chunk length in samples.
#     hop_samples = int(chunk_samples * (1 - overlap_ratio))  # hop size between chunks.
    
#     start_sample = 0
#     while start_sample < total_samples:
#         end_sample = min(start_sample + chunk_samples, total_samples)
#         chunk_wav = wav[:, :, start_sample:end_sample]  # shape: [1, channels, chunk_samples]
        
#         with torch.no_grad():
#             # Encode the chunk to get discrete tokens.
#             prompt_tokens, _ = model.compression_model.encode(chunk_wav)
#             # prompt_tokens shape: [1, K, S] (K: number of codebooks, S: number of tokens for this chunk)
            
#             # Get the intermediate hidden state representation using the helper function.
#             hidden_states_dict = get_hidden_states_from_tokens(
#                 model,
#                 prompt_tokens,
#                 text='a song',
#                 layer_indices=layer_indices
#             )
#             # hidden_chunk shape: [1, S, D]
        
#         example_layer = layer_indices[0]  # pick the first
#         hidden_chunk = hidden_states_dict[example_layer]
#         S = hidden_chunk.shape[1]
        
#         # Determine the start time (in seconds) of the chunk and map it to a frame index.
#         chunk_start_time = start_sample / sr
#         chunk_start_frame = int(round(chunk_start_time * fps))
        
#         # Clip the chunk if it exceeds the global duration.
#         if chunk_start_frame + S > global_frames:
#             S = global_frames - chunk_start_frame
#             for idx in layer_indices:
#                 hidden_states_dict[idx] = hidden_states_dict[idx][:, :S, :]
        
#         #  Accumulate each layer's hidden states
#         for idx in layer_indices:
#             global_hidden_dict[idx][:, chunk_start_frame:chunk_start_frame + S, :] += hidden_states_dict[idx]
#         global_weights[:, chunk_start_frame:chunk_start_frame + S, :] += 1.0
        
#         # Move the window.
#         start_sample += hop_samples
    
#     # Average overlapping frames
#     for idx in layer_indices:
#         global_hidden_dict[idx] /= global_weights

#     # Now resample each layer's hidden state
#     final_dict = {}
#     for idx in layer_indices:
#         # [1, T, D]
#         hs = global_hidden_dict[idx]
#         hs_resampled = resample_hidden_states(hs, model, frame_length)
#         final_dict[idx] = hs_resampled.squeeze(0)  # remove batch dim: [T', D]

#     return final_dict

def main():
    # Example usage
    model = get_musicgen_model("small", device='cpu')
    filename =  get_filenames()[0]
    dir = "./data/processed/"
    max_chunk_length = 5.0 # seconds
    frame_length = 4096 / 44100  # ~0.093 seconds

    result = extract_song_hidden_representation(
        filename,
        dir,
        model,
        max_chunk_length,
        frame_length
    )
    
    print(result)

if __name__ == "__main__":
    main()