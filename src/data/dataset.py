import autorootcwd
import os
import random
from collections import defaultdict
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split

from src.utils import (
    pitch_shift_cqt,
    get_torch_device,
    get_split_filenames,
    get_chord_quality,
    get_synthetic_filenames,
    get_split_synthetic_filenames,
    get_chord_annotation,
    transpose_chord_id_vector,
    SMALL_VOCABULARY,
    NUM_CHORDS,
    HOP_LENGTH,
    SR,
    BINS_PER_OCTAVE,
)
from src.data.beats.beatwise_resampling import (
    resample_features_by_beat,
    get_beatwise_chord_annotation,
    get_resampled_full_beats,
)


class FullChordDataset(Dataset):
    def __init__(
        self,
        filenames: List[str] = None,
        hop_length: int = HOP_LENGTH,
        mask_X: bool = False,
        input_dir: str = "./data/processed",
        gen_reduction: str = "concat",
        gen_model_name: str = "large",
        small_vocab: bool = SMALL_VOCABULARY,
        spectrogram_type: str = "cqt",
        use_augs: bool = False,
        dev_mode: bool = False,
        input_transitions: bool = False,
        beat_wise_resample: bool = False, 
        beat_resample_interval: float = 1,
        perfect_beat_resample: bool = False,
        synthetic_filenames: List[str] = None,
        synthetic_input_dir: str = "./data/synthetic",
        synthetic_only: bool = False,
    ):
        """
        Initialize a chord dataset. Each sample is a tuple of features and chord annotation.
        Args:
            filenames (list) = None: A list of filenames to include in the dataset. If None, all files in the processed audio directory are included.
            hop_length (int): The hop length used to compute the log CQT.
            generative_features (bool): If True, the dataset loads generative features from MusicGen.
            mask_X (bool): If True, the dataset masks the class label X as -1 to be ignored by the loss function.
            input_dir (str): The directory where the audio files are stored.
            small_vocab (bool): If True, the dataset uses a small vocabulary of chords.
            gen_reduction (str): The reduction method to use for the generative features. Options are 'concat', 'avg', 'codebook_0', 'codebook_1', 'codebook_2', 'codebook_3'.
            gen_model_name (str): The size of the generative model to use. Options are 'large' or 'small'.
            input_transitions (bool): If True, the dataset uses transitions in the input features.
            spectrogram_type (str): The type of spectrogram to use. Options are 'cqt', 'chroma', 'linear' or 'mel'.
            use_augs (bool): If True, the dataset uses augmented CQT and chord annotation files.
            dev_mode (bool): If True, we ignore generative features to allow for dataset use for analysis.
            beat_wise_resample (bool): If True, resample all modalities by beat.
            beat_resample_interval (float): The beat interval to use for resampling.
            perfect_beat_resample (bool): If True, resample all modalities by beat using the perfect beat annotation provided by the labels.
        """
        if filenames is None:
            print("Using all filenames!")
            filenames = os.listdir(f"{input_dir}/audio")
            # Filter out non-mp3 files and remove the extension
            filenames = [
                filename.split(".")[0]
                for filename in filenames
                if filename.endswith(".mp3")
            ]
        self.input_dir = input_dir
        self.filenames = filenames
        self.hop_length = hop_length
        self.mask_X = mask_X
        self.dev_mode = dev_mode
        self.input_transitions = input_transitions
        self.gen_reduction = gen_reduction
        self.gen_model_name = gen_model_name
        assert self.gen_reduction in [
            "avg",
            "concat",
            "codebook_0",
            "codebook_1",
            "codebook_2",
            "codebook_3",
            None,
        ], f"Invalid reduction: {self.gen_reduction}. Must be 'concat' or 'codebook_3'."

        assert not (
            self.input_transitions and beat_wise_resample
        ), "Cannot have input transitions and beat-wise resampling at the same time."

        dir_mapping = {
            "cqt": "cqts",
            "chroma": "chroma_cqts",
            "linear": "linear",
            "mel": "mels",
        }
        feature_dir = dir_mapping.get(spectrogram_type, "cqts")

        self.use_augs = use_augs
        self.feature_cache_dir = (
            f"{self.input_dir}/cache/{self.hop_length}/{feature_dir}"
        )
        self.gen_cache_dir = f"{self.input_dir}/cache/{self.hop_length}/gen-{self.gen_model_name}/{self.gen_reduction}"
        self.chord_cache_dir = f"{self.input_dir}/cache/{self.hop_length}/chords"
        self.small_vocab = small_vocab
        if self.small_vocab:
            self.chord_cache_dir = (
                f"{self.input_dir}/cache/{self.hop_length}/chords_small_vocab"
            )
        if self.use_augs:
            self.aug_cqt_cache_dir = f"{self.feature_cache_dir}/augs"
            self.aug_chord_cache_dir = f"{self.chord_cache_dir}/augs"
            self.aug_gen_cache_dir = f"{self.gen_cache_dir}/augs"

        self.beat_wise_resample = beat_wise_resample
        self.beat_resample_interval = beat_resample_interval
        self.perfect_beat_resample = perfect_beat_resample

        self.synthetic = synthetic_filenames is not None
        if self.synthetic:
            self.synthetic_filenames = synthetic_filenames
            self.synthetic_input_dir = synthetic_input_dir
            self.synthetic_cqt_dir = f"{self.synthetic_input_dir}/cache/{self.hop_length}/{feature_dir}"      
            self.synthetic_chord_dir = f"{self.synthetic_input_dir}/cache/{self.hop_length}/chords"
            self.synthetic_gen_cache_dir = f"{self.synthetic_input_dir}/cache/{self.hop_length}/gen-{self.gen_model_name}/{self.gen_reduction}"
            self.synthetic_only = synthetic_only
            if synthetic_only:
                # If synthetic_only is True, we ignore the original dataset
                self.filenames = []
                

    def __len__(self):
        length = len(self.filenames) 
        if self.synthetic:
            length += len(self.synthetic_filenames)
        return length

    def get_transitions(self, idx):
        """
        Get the transitions binary vector for the given index.
        Args:
            idx (int): The index of the item to get.
        Returns:
            A tensor of transitions.
        """
        filename = self.get_filename(idx)
        _, transitions = get_chord_annotation(
            filename, frame_length=self.hop_length / SR, return_transitions=True, override_dir=f"{self.input_dir}/chords"
        )
        return transitions.long()

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        cqt, gen, chord_ids = self.__get_cached_item(idx)

        if self.mask_X:
            chord_ids = torch.where(chord_ids == 1, -1, chord_ids)

        return self.get_minimum_length_frame(cqt, gen, chord_ids)

    def get_aug_item(self, idx, pitch_aug: int) -> Tuple[Tensor, Tensor]:
        cqt, gen, chord_ids = self.__get_cached_item(idx, pitch_aug)
        if self.mask_X:
            chord_ids = torch.where(chord_ids == 1, -1, chord_ids)

        return self.get_minimum_length_frame(cqt, gen, chord_ids)

    def get_minimum_length_frame(
        self, *tensors: Optional["Tensor"]
    ) -> Tuple[Optional["Tensor"], ...]:
        """
        Returns each non-None tensor trimmed to the minimum length along the first dimension
        among all provided non-None tensors, and returns None in slots corresponding to None inputs.

        Args:
            *tensors (Optional[Tensor]): An arbitrary number of tensors (or None).

        Returns:
            Tuple[Optional[Tensor], ...]: A tuple of tensors, each sliced to the minimum length
            (if the original input is a tensor), or None (if the original input was None).
        """

        EMPTY_TENSOR = torch.empty(0)
        valid_tensors = [t for t in tensors if t is not None]
        if not valid_tensors:
            return (EMPTY_TENSOR, EMPTY_TENSOR, EMPTY_TENSOR)

        min_length = min(t.shape[0] for t in valid_tensors)
        return tuple(t[:min_length] if t is not None else EMPTY_TENSOR for t in tensors)
    
    def get_class_weights_old(self, epsilon=10, alpha=0.3) -> torch.Tensor:
        """
        DEPRECATED: Calculate chord loss weights for the dataset.

        Args:
            epsilon (float): A value to prevent division by zero. The effective number of samples for each class is epsilon + counts.
            alpha (float): The exponent of the inverse frequency. The higher the value, the more the weights are skewed towards the rare classes. Should be in [0,1].

        Returns:
            weights (Tensor): The chord loss weights, normalized.
        """
        all_chord_ids = torch.cat(
            [self[i][2].flatten() for i in range(len(self))]
        )  # Flatten all chord IDs
        all_chord_ids = all_chord_ids[all_chord_ids != -1]  # Remove -1 (ignored labels)

        all_chord_ids = all_chord_ids.long()

        counts = torch.bincount(all_chord_ids, minlength=NUM_CHORDS).float()

        weights = 1.0 / (counts + epsilon) ** alpha  # Inverse frequency weights

        # Mask zero-count classes
        nonzero_mask = counts > 0  # Only consider seen classes
        weights[~nonzero_mask] = 0  # Set weights of unseen classes to 0

        # Compute normalization factor using only nonzero classes
        scaling_factor = (counts[nonzero_mask] * weights[nonzero_mask]).sum() / counts[
            nonzero_mask
        ].sum()
        weights /= scaling_factor  # Normalize

        if self.mask_X:
            weights[1] = 0  # Ensure class '1' has zero weight if it is masked

        return weights
    

    def get_class_counts(self, aug_shift_prob=None) -> torch.Tensor:
        """
        Calculate chord counts for the dataset. This method is used to compute the class weights. It calculates expected counts, accounting for any pitch shifting probabilities.
        Args:
            aug_shift_prob (float): Probability of applying a pitch shift.
        Returns:
            counts (torch.Tensor): The chord counts for each class.
        """
        if aug_shift_prob is None or not self.use_augs:
            aug_shift_prob = 0
        # Allowed nonzero pitch shifts.
        low, high = -5, 6
        allowed_shifts = [s for s in range(low, high + 1) if s != 0]

        total_counts = torch.zeros(NUM_CHORDS, dtype=torch.float)

        for i in range(len(self.filenames)):
            # Load the chord IDs for the current sample. Frame-based are good enough estimates based on duration.
            chord_ids = torch.load(
                f"{self.chord_cache_dir}/{self.filenames[i]}.pt",
                weights_only=True,
            ).flatten()
            chord_ids = chord_ids[chord_ids != -1]
            if chord_ids.numel() == 0:
                continue
            # Count chords in the original (unshifted) annotation.
            orig_count = torch.bincount(chord_ids, minlength=NUM_CHORDS).float()
            expected_count = (1 - aug_shift_prob) * orig_count
            # For each allowed pitch shift, transpose the chord IDs and add the weighted counts.
            for s in allowed_shifts:
                transposed = transpose_chord_id_vector(chord_ids, s)
                transposed = torch.tensor(transposed, dtype=torch.long, device=chord_ids.device)
                transposed = transposed.flatten()
                transposed = transposed[transposed != -1]
                if transposed.numel() == 0:
                    continue
                trans_count = torch.bincount(transposed, minlength=NUM_CHORDS).float()
                expected_count += (aug_shift_prob / len(allowed_shifts)) * trans_count

            total_counts += expected_count
        
        # Account for synthetic data
        if self.synthetic:
            for i in range(len(self.synthetic_filenames)):
                # Load the chord IDs for the synthetic sample (no augmentation).
                chord_ids = torch.load(
                    f"{self.synthetic_chord_dir}/{self.synthetic_filenames[i]}.pt",
                    weights_only=True,
                ).flatten()
                chord_ids = chord_ids[chord_ids != -1]
                if chord_ids.numel() == 0:
                    continue
                # Just do direct counts
                synth_count = torch.bincount(chord_ids, minlength=NUM_CHORDS).float()
                total_counts += synth_count
        
        return total_counts

    
    def get_class_weights(
        self,
        epsilon: float = 10,
        alpha: float = 0.3,
        aug_shift_prob=None
    ) -> torch.Tensor:
        """
        Calculate chord loss weights while accounting for pitch shifting augmentation.
        For each sample, the method computes:
          - (1 - aug_shift_prob) * counts(original chord vector) +
          - aug_shift_prob * (1 / N_shifts) * sum(counts(transposed chord vector) for each allowed shift)
        The inverse frequency weights are computed from these aggregated counts.
        
        Args:
            aug_shift_prob (float): Probability of applying a pitch shift.
            pitch_range (tuple): Inclusive bounds for allowed semitone shifts (zero is excluded),
                                 e.g. (-5, 6) corresponds to shifts in {-5, -4, ..., -1, 1, ..., 6}.
            epsilon (float): Small value to avoid division by zero.
            alpha (float): Exponent for inverse frequency weighting.
            
        Returns:
            weights (torch.Tensor): Normalized inverse frequency weights for each chord class.
        """

        total_counts = self.get_class_counts(aug_shift_prob=aug_shift_prob)

        # Compute inverse frequency weights.
        weights = 1.0 / (total_counts + epsilon) ** alpha
        nonzero_mask = total_counts > 0
        # Zero out weights for classes that never appear.
        weights[~nonzero_mask] = 0
        if nonzero_mask.any():
            scaling_factor = (total_counts[nonzero_mask] * weights[nonzero_mask]).sum() / total_counts[nonzero_mask].sum()
            weights /= scaling_factor

        if self.mask_X:
            weights[1] = 0

        return weights
    
    def is_synthetic(self, idx):
        """
        Check if the given index corresponds to a synthetic file.
        Args:
            idx (int): The index of the item to check.
        Returns:
            bool: True if the index corresponds to a synthetic file, False otherwise.
        """
        return idx >= len(self.filenames) and idx < len(self.filenames) + len(self.synthetic_filenames)

    def __get_cached_item(self, idx, pitch_aug=None) -> Tuple[Tensor, Tensor]:
        """
        Get a cached item from the dataset.
        Args:
            idx (int): The index of the item to get.
            aug (str): The pitch augmentation to retrieve from.
        Returns:
            A tuple of the CQT, generative features, and chord IDs.
        """
        
        aug = pitch_aug is not None and pitch_aug != 0 and self.use_augs

        if self.is_synthetic(idx):
            filename = self.synthetic_filenames[idx - len(self.filenames)]
            cqt_dir = self.synthetic_cqt_dir
            chord_dir = self.synthetic_chord_dir
            gen_dir = self.synthetic_gen_cache_dir
        else:
            filename = self.filenames[idx]
            cqt_dir = self.feature_cache_dir
            chord_dir = self.chord_cache_dir
            gen_dir = self.gen_cache_dir
            if aug:
                # Only use the augmented CQT and chord files for non-synthetic files
                cqt_dir = self.aug_cqt_cache_dir
                chord_dir = self.aug_chord_cache_dir
                gen_dir = self.aug_gen_cache_dir
                filename = f"{filename}_shifted_{pitch_aug}"

        cqt = torch.load(f"{cqt_dir}/{filename}.pt", weights_only=True, map_location=get_torch_device())
        chord_ids = torch.load(f"{chord_dir}/{filename}.pt", weights_only=True, map_location=get_torch_device())
        try:
            gen = torch.load(
                f"{gen_dir}/{filename}.pt",
                weights_only=True,
                map_location=get_torch_device(),
            )
        except FileNotFoundError:
            # If the generative features are not found, use None. It is later converted to an empty tensor.
            if self.dev_mode or self.gen_reduction is None:
                gen = None
            else:
                raise FileNotFoundError(
                    f"Generative features not found for {filename}. Please run the generative feature extraction script."
                )

        # If beat-wise resampling is enabled, resample all modalities:
        if self.beat_wise_resample:
            # Resample features using your previously defined torch-compatible function:
            song_end = (cqt.shape[0]+1) * self.hop_length / SR
            cqt = resample_features_by_beat(
                features=cqt,
                filename=filename,
                beat_interval=self.beat_resample_interval,
                perfect_beat_resample=self.perfect_beat_resample,
                frame_rate=SR / self.hop_length,
                override_dir_chord=f"{self.input_dir}/chords",
                override_dir_beat=f"{self.input_dir}/beats",
                song_end=song_end,
            )
            # Do the same for generative features if available:
            if gen is not None and gen.shape[0] > 0:
                gen = resample_features_by_beat(
                    features=gen,
                    filename=filename,
                    beat_interval=self.beat_resample_interval,
                    perfect_beat_resample=self.perfect_beat_resample,
                    frame_rate=SR / self.hop_length,
                    override_dir_chord=f"{self.input_dir}/chords",
                    override_dir_beat=f"{self.input_dir}/beats",
                    song_end=song_end,
                )
            # Resample chords:
            chord_ids = get_beatwise_chord_annotation(
                filename,
                beat_interval=self.beat_resample_interval,
                perfect_beat_resample=self.perfect_beat_resample,
                override_dir_chord=f"{self.input_dir}/chords",
                override_dir_beat=f"{self.input_dir}/beats",
                song_end=song_end,
            )

        if self.input_transitions:
            # Get transitions for the input features
            transitions = self.get_transitions(idx)  # Shape: (frames)
            cqt, transitions = self.get_minimum_length_frame(cqt, transitions)
            # Concatenate transitions to the CQT
            cqt = torch.cat((cqt, transitions.unsqueeze(1)), dim=1)

        return cqt, gen, chord_ids

    def get_beats(self, idx):
        """
        Get the 'beat times' for the features provided by this class. If beat-wise resampling is enabled, we use the actual beats from the song. Otherwise, return the 'beats' from the CQT.
        Args:
            idx (int): The index of the song.
        Returns:
            A list of beat times.
        """
        filename = self.get_filename(idx)
        if self.beat_wise_resample:
            # Get raw cqt to find song length for beat-wise cutoff
            aug = self.use_augs and self.aug_cqt_cache_dir is not None
            cqt_dir = self.feature_cache_dir if not aug else self.aug_cqt_cache_dir
            song_end = (
                torch.load(f"{cqt_dir}/{filename}.pt", weights_only=True).shape[0] + 1
            ) * self.hop_length / SR

            # If beat-wise resampling is enabled, use the resampled beat times.
            return get_resampled_full_beats(
                filename=filename,
                beat_interval=self.beat_resample_interval,
                perfect_beat_resample=self.perfect_beat_resample,
                override_dir_beat=f"{self.input_dir}/beats",
                override_dir_chord=f"{self.input_dir}/chords",
                song_end=song_end
            )
        else:
            # Otherwise, 'beats' are the CQT frames.
            cqt = self[idx][0]
            # Get the beat times from the CQT. [0, hop_length, 2*hop_length, ..., n * hop_length]
            beats = torch.arange(cqt.shape[0] + 1) * self.hop_length / SR
            return beats.tolist()

    def get_filename(self, idx):
        if self.is_synthetic(idx):
            return self.synthetic_filenames[idx - len(self.filenames)]
        return self.filenames[idx]


class FixedLengthRandomChordDataset(Dataset):
    """
    A chord dataset that returns fixed length frames, where the frame location in each sample of audio is uniformly random.
    """

    def __init__(
        self,
        cqt_pitch_shift=False,
        audio_pitch_shift=False,
        aug_shift_prob=0.5,
        segment_length=10,
        filenames=None,
        hop_length=HOP_LENGTH,
        mask_X=False,
        gen_reduction="concat",
        gen_model_name="large",
        spectrogram_type="cqt",
        input_transitions=False,
        input_dir="./data/processed/",
        beat_wise_resample=False,
        beat_resample_interval=1,
        perfect_beat_resample=False,
        synthetic_filenames=None,
        synthetic_input_dir="./data/synthetic_data",
        synthetic_only=False,
    ):
        """
        Initialize a chord dataset. Each sample is a tuple of features and chord annotation.
        Args:
            random_pitch_shift (bool): If True, the dataset randomly shifts the pitch of the cqt.
            filenames (list) = None: A list of filenames to include in the dataset. If None, all files in the processed audio directory are included
            segment_length (int): The length of the segment in seconds.
            hop_length (int): The hop length used to compute the log CQT.
            mask_X (bool): If True, the dataset masks the class label X as -1 to be ignored by the loss function.
        """
        super().__init__()
        self.full_dataset = FullChordDataset(
            filenames=filenames,
            hop_length=hop_length,
            mask_X=mask_X,
            gen_reduction=gen_reduction,
            gen_model_name=gen_model_name,
            input_dir=input_dir,
            use_augs=audio_pitch_shift,
            input_transitions=input_transitions,
            spectrogram_type=spectrogram_type,
            beat_wise_resample=beat_wise_resample,
            beat_resample_interval=beat_resample_interval,
            perfect_beat_resample=perfect_beat_resample,
            synthetic_filenames=synthetic_filenames,
            synthetic_input_dir=synthetic_input_dir,
            synthetic_only=synthetic_only,
        )
        self.audio_pitch_shift = audio_pitch_shift
        self.aug_shift_prob = aug_shift_prob
        self.cqt_pitch_shift = cqt_pitch_shift
        self.segment_length = segment_length

    def get_random_shift(self, lower: int = -5, upper: int = 6) -> int:
        """
        Returns 0 with probability 1 - aug_shift_prob.
        Returns a random value between lower and upper bounds inclusive \ {0} with probability aug_shift_prob.
        """
        if random.random() < self.aug_shift_prob:
            shift = random.randint(lower, upper)
            while shift == 0:
                shift = random.randint(lower, upper)
            return shift
        else:
            return 0

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:

        shift = 0
        if self.audio_pitch_shift:
            shift = self.get_random_shift()

        if shift == 0:
            (
                full_cqt,
                gen_features,
                full_chord_ids,
            ) = self.full_dataset[idx]
        else:
            full_cqt, gen_features, full_chord_ids = self.full_dataset.get_aug_item(
                idx, shift
            )

        # Trim the features to a given sample length
        if self.full_dataset.beat_wise_resample:
            # Get beat timestamps (in seconds) for this song
            beats = self.full_dataset.get_beats(idx)
            song_duration = beats[-1]
            # Sample a random starting time such that [t_start, t_start + segment_length] lies within the song.
            t_max = max(
                song_duration - self.segment_length, 0
            )  # Clip to avoid negative values
            t_start = random.uniform(0, t_max)
            t_end = t_start + self.segment_length

            # Select all beat indices with some overlap with the segment [t_start, t_end].
            beat_indices = [
                i
                for i in range(len(beats) - 1)
                if max(beats[i], t_start) < min(beats[i + 1], t_end)
            ]
            if not beat_indices:
                raise ValueError(
                    f"No beats found between {t_start:.2f} and {t_end:.2f} seconds in song {self.full_dataset.get_filename(idx)}."
                )

            # Slice the beat-synchronous features using these beat indices.
            cqt_patch = full_cqt[beat_indices, :]
            # If not the empty tensor
            if gen_features is not None and gen_features.shape[0] > 0:
                gen_features_patch = gen_features[beat_indices, :]
            else:
                gen_features_patch = torch.empty(0)
            chord_ids_patch = full_chord_ids[beat_indices]
        else:
            # Use the standard method based on frames
            segment_length_samples = int(
                self.segment_length * (SR / self.full_dataset.hop_length)
            )
            if full_cqt.shape[0] > segment_length_samples:
                start_idx = torch.randint(
                    0, full_cqt.shape[0] - segment_length_samples, (1,)
                ).item()
                cqt_patch = full_cqt[start_idx : start_idx + segment_length_samples]
                gen_features_patch = gen_features[
                    start_idx : start_idx + segment_length_samples
                ]
                chord_ids_patch = full_chord_ids[
                    start_idx : start_idx + segment_length_samples
                ]
            else:
                pad_length = segment_length_samples - full_cqt.shape[0]
                cqt_patch = torch.cat(
                    [
                        full_cqt,
                        torch.zeros(
                            (pad_length, full_cqt.shape[1]), device=full_cqt.device
                        ),
                    ]
                )
                if self.full_dataset.gen_reduction is not None:
                    gen_features_patch = torch.cat(
                        [
                            gen_features,
                            torch.zeros(
                                (pad_length, gen_features.shape[1]),
                                device=gen_features.device,
                            ),
                        ]
                    )
                else:
                    gen_features_patch = torch.empty(0)
                chord_ids_patch = torch.cat(
                    [
                        full_chord_ids,
                        torch.full((pad_length,), -1, device=full_chord_ids.device),
                    ]
                )

        # if self.cqt_pitch_shift:
            shift = self.get_random_shift()
            if shift != 0:
                cqt_patch = pitch_shift_cqt(cqt_patch, shift, BINS_PER_OCTAVE)
                chord_ids_patch = torch.tensor(
                    transpose_chord_id_vector(chord_ids_patch.cpu(), shift), dtype=torch.long
                ).to(cqt_patch.device)

        return cqt_patch, gen_features_patch, chord_ids_patch

    def __len__(self):
        return len(self.full_dataset)


class FixedLengthChordDataset(Dataset):
    """
    A chord dataset that returns fixed length frames, that splits up the audio in a test set into fixed length frames.
    """

    def __init__(
        self,
        segment_length=10,
        filenames=None,
        hop_length=HOP_LENGTH,
        mask_X=False,
        gen_reduction="concat",
        gen_model_name="large",
        spectrogram_type="cqt",
        input_transitions=False,
        input_dir="./data/processed/",
        beat_wise_resample=False,
        beat_resample_interval=1,
        perfect_beat_resample=False,
        synthetic_filenames=None,
        synthetic_input_dir="./data/synthetic_data",
    ):
        """
        Creates an instance of the FixedLengthChordDataset class.

        Args:
            full_length_dataset (FullChordDataset): The full chord dataset.
            segment_length (int): The length of the segment in seconds.
            hop_length (int): The hop length used to compute the log CQT.
            cached (bool): If True, the dataset loads cached CQT and chord annotation files. If False, the CQT and chord annotation are computed on the fly.
            filenames (list): A list of filenames to include in the dataset. If None, all files in the processed audio directory are included.
        """
        super().__init__()
        self.full_dataset = FullChordDataset(
            filenames=filenames,
            hop_length=hop_length,
            mask_X=mask_X,
            input_dir=input_dir,
            gen_reduction=gen_reduction,
            gen_model_name=gen_model_name,
            spectrogram_type=spectrogram_type,
            input_transitions=input_transitions,
            beat_wise_resample=beat_wise_resample,
            beat_resample_interval=beat_resample_interval,
            perfect_beat_resample=perfect_beat_resample,
            synthetic_filenames=synthetic_filenames,
            synthetic_input_dir=synthetic_input_dir,
        )
        self.segment_length = segment_length
        self.data = self.generate_fixed_segments()

    def generate_fixed_segments(self) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate fixed length segments from the full dataset. Each song is split into segments of the specified length.

        Args:
            full_dataset (FullChordDataset): The full chord dataset.
            frame_length (int): The length of the frame in seconds.

        Returns:
            data (list): A list of tuples, where each tuple is a fixed length frame of features and chord annotation.
        """
        data = []
        # If beat-wise resampling is not enabled, use the frame-based segmentation as before.
        if not self.full_dataset.beat_wise_resample:
            self.segment_length_samples = int(
                self.segment_length * SR / self.full_dataset.hop_length
            )
            for i in range(len(self.full_dataset)):
                cqt, gen, chord_ids = self.full_dataset[i]
                # Loop over each segment using the fixed frame length.
                for j in range(0, cqt.shape[0], self.segment_length_samples):
                    data.append(
                        (
                            cqt[j : j + self.segment_length_samples],
                            gen[j : j + self.segment_length_samples],
                            chord_ids[j : j + self.segment_length_samples],
                        )
                    )
            return data
        else:
            # For beat-wise resampling we approximately partition the song into segments
            for i in range(len(self.full_dataset)):
                cqt, gen, chord_ids = self.full_dataset[i]
                # Get the beat boundaries for this song.
                beats = self.full_dataset.get_beats(
                    i
                )  # Assumed to be a list of times (in seconds)
                total_duration = beats[-1]
                # Decide how many segments you want; segment_length is a guideline (in seconds).
                num_segments = max(
                    1, int(np.ceil(total_duration / self.segment_length))
                )

                # Partition the B beat intervals evenly into num_segments segments.
                B = len(beats) - 1
                boundaries = np.linspace(0, B, num_segments + 1, dtype=int)
                # Now generate segments that cover the whole song without overlapping beats.
                for j in range(num_segments):
                    start_idx = boundaries[j]
                    end_idx = boundaries[j + 1]
                    if end_idx <= start_idx:
                        continue  # Safeguard.
                    cqt_seg = cqt[start_idx:end_idx]
                    if gen is not None:
                        gen_seg = gen[start_idx:end_idx]
                    else:
                        gen_seg = None
                    chord_seg = chord_ids[start_idx:end_idx]
                    data.append((cqt_seg, gen_seg, chord_seg))
            return data

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get the original data from the dataset
        data = self.dataset[index]
        # Return both the data and the index
        return data, index


def generate_datasets(
    train_split: str,
    input_dir: str,
    segment_length: int,
    mask_X: bool,
    hop_length: int,
    cqt_pitch_shift: bool = False,
    audio_pitch_shift: bool = False,
    aug_shift_prob: float = 0.9,
    input_transitions: bool = False,
    gen_reduction: str = "codebook_3",
    gen_model_name: str = "large",
    spectrogram_type: str = "cqt",
    beat_wise_resample: bool = False,
    beat_resample_interval: float = 1,
    perfect_beat_resample: bool = False,
    perfect_beat_resample_eval: bool = False,
    subset_size: int =None,
    use_synthetic: bool =False,
    synthetic_ratio: float = 1,
    synthetic_input_dir:str=None,
    synthetic_only: bool = False,
    test_on_synthetic: bool = False,
):
    """
    Generate the training, validation, and test datasets.

    Args:
        train_filenames (list): The list of training filenames.
        val_filenames (list): The list of validation filenames.
        test_filenames (list): The list of test filenames.
        input_dir (str): The directory where the audio files are stored.
        segment_length (int): The length of the segment in seconds.
        mask_X (bool): If True, the dataset masks the class label X as -1 to be ignored by the loss function.
        hop_length (int): The hop length used to compute the log CQT.
        subset_size (int): The size of the subset to use. If None, the full dataset is used.

    Returns:
        train_dataset (FixedLengthRandomChordDataset): The training dataset.
        val_dataset (FixedLengthChordDataset): The validation dataset.
        test_dataset (FullChordDataset): The test dataset.
        val_final_test_dataset (FullChordDataset): The validation dataset for the final test.
    """
    train_filenames, val_filenames, test_filenames = get_split_filenames(input_dir)

    if train_split == "80":
        train_filenames = train_filenames + val_filenames
        val_filenames = []
    elif train_split == "100":
        train_filenames = train_filenames + val_filenames + test_filenames
        val_filenames = []
        test_filenames = []

    if subset_size:
        train_filenames = train_filenames[:subset_size]
        val_filenames = val_filenames[:subset_size]
        test_filenames = test_filenames[:subset_size]

    if use_synthetic:
        synthetic_filenames_train, _, _ = get_split_synthetic_filenames(synthetic_input_dir)
        # Use the synthetic filenames for training and validation
        synthetic_filenames_train = synthetic_filenames_train[:int(len(train_filenames) * synthetic_ratio)]
    else:
        synthetic_filenames_train = None

    if test_on_synthetic:
        _, _, test_synthetic_filenames = get_split_synthetic_filenames(synthetic_input_dir)
    else:
        test_synthetic_filenames = None

    train_dataset = FixedLengthRandomChordDataset(
        filenames=train_filenames,
        segment_length=segment_length,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        cqt_pitch_shift=cqt_pitch_shift,
        audio_pitch_shift=audio_pitch_shift,
        aug_shift_prob=aug_shift_prob,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample,
        synthetic_filenames=synthetic_filenames_train,
        synthetic_input_dir=synthetic_input_dir,
        synthetic_only=synthetic_only,
    )
    val_dataset = FixedLengthChordDataset(
        filenames=val_filenames,
        segment_length=segment_length,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample_eval,
    )
    test_dataset = FullChordDataset(
        filenames=test_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample_eval,
    )
    train_final_test_dataset = FullChordDataset(
        filenames=train_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample,
    )
    val_final_test_dataset = FullChordDataset(
        filenames=val_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample_eval,
    )
    synthetic_final_test_dataset = FullChordDataset(
        filenames=[],
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        gen_reduction=gen_reduction,
        gen_model_name=gen_model_name,
        spectrogram_type=spectrogram_type,
        input_transitions=input_transitions,
        beat_wise_resample=beat_wise_resample,
        beat_resample_interval=beat_resample_interval,
        perfect_beat_resample=perfect_beat_resample_eval,
        synthetic_filenames=test_synthetic_filenames,
        synthetic_input_dir=synthetic_input_dir,
        synthetic_only=True
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_final_test_dataset,
        val_final_test_dataset,
        synthetic_final_test_dataset,
    )


def get_calibrated_priors(
    training_dataset: FullChordDataset,
    target_dataset: FullChordDataset,
    aug_shift_prob: float = 0.5,
    smoothing: float = 1e-6,
    root_invariance: bool = True,
    return_as_log: bool = True,
) -> torch.Tensor:
    """
    Compute ratio-based calibration priors for chord recognition.
    
    Specifically:
      ratio[y] = ( P_target(y) / P_train(y) )

    If root_invariance=True, then instead of having a unique ratio for each chord ID, 
    we group chords by their "quality" (e.g., 'maj','min','dim', etc.), sum up the 
    probabilities P_train and P_target within each quality, and assign a single ratio 
    to all chord IDs in that quality group.

    Args:
        training_dataset (FullChordDataset): The dataset used in training.
        target_dataset (FullChordDataset): The dataset whose distribution we want to match 
                                           (often a real subset or validation set).
        aug_shift_prob (float): Probability of pitch-shifting used for both training_dataset
                                and target_dataset when counting chords. Adjust if needed.
        smoothing (float): A tiny constant to avoid division by zero.
        root_invariance (bool): If True, lumps all chords with the same quality together 
                                so they share the same ratio.

    Returns:
        ratio (torch.Tensor): A 1D tensor of length NUM_CHORDS, 
                              where ratio[y] = P_target(y) / P_train(y)
                              (possibly collapsed over roots if root_invariance=True).
    """
    # Compute distributions on train & target sets
    device = get_torch_device()
    p_train = training_dataset.get_class_counts(aug_shift_prob=aug_shift_prob)
    p_target = target_dataset.get_class_counts(aug_shift_prob=aug_shift_prob)

    # Normalize each distribution
    train_sum = p_train.sum()
    target_sum = p_target.sum()

    if train_sum < smoothing:
        raise ValueError("Training dataset chord counts sum to zero. Check data or smoothing.")
    if target_sum < smoothing:
        raise ValueError("Target dataset chord counts sum to zero. Check data or smoothing.")

    p_train = p_train / train_sum
    p_target = p_target / target_sum

    # If not root-invariant, return elementwise ratio
    ratio = (p_target + smoothing) / (p_train + smoothing)
    if not root_invariance:
        if return_as_log:
            return torch.log(torch.tensor(ratio)).to(device)
        else:
            return torch.tensor(ratio).to(device)

    # Root invariance, group chords by 'quality' & average over roots
    chord_ids_by_quality = defaultdict(list)

    # Build a dictionary mapping from chord quality -> list of chord IDs
    for chord_id in range(len(ratio)):
        quality_str = get_chord_quality(
            chord_id, 
            use_small_vocab=training_dataset.small_vocab,
            return_idx=False
        )
        chord_ids_by_quality[quality_str].append(chord_id)

    # Sum up p_train and p_target for each quality
    sum_train_by_quality = {}
    sum_target_by_quality = {}

    for quality, ids in chord_ids_by_quality.items():
        sum_train_by_quality[quality] = p_train[ids].sum()
        sum_target_by_quality[quality] = p_target[ids].sum()

    # Compute a single ratio for each quality
    ratio_by_quality = {}
    for quality in chord_ids_by_quality:
        ratio_by_quality[quality] = (
            sum_target_by_quality[quality] + smoothing
        ) / (
            sum_train_by_quality[quality] + smoothing
        )

    # Assign that ratio back to every chord ID in that quality group
    ratio_root_inv = torch.zeros_like(ratio)
    for quality, ids in chord_ids_by_quality.items():
        ratio_root_inv[ids] = ratio_by_quality[quality]

    if return_as_log:
        return torch.log(ratio_root_inv).to(device)
    else:
        return ratio_root_inv.to(device)

def main():
    import torch

    torch.manual_seed(0)
    filenames = os.listdir("./data/processed/audio")

    # Filter out non-mp3 files and remove the extension
    filenames = [
        filename.split(".")[0] for filename in filenames if filename.endswith(".mp3")
    ]

    # Create a dataset
    dataset = FullChordDataset(filenames)

    # Split the dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Number of training samples: {len(train_dataset)}")

    # Print first item with and without masking and then counts of 1s in each
    idx = 100
    print("Without masking")
    print(dataset[idx][1])
    print(torch.sum(dataset[idx][1] == 1))

    print("With masking")
    print(dataset[idx][1])
    print(torch.sum(dataset[idx][1] == 1))

    print("Test Calc Chord Loss Weights")
    weights = dataset.get_class_weights()
    print(weights)
    print(sum(weights))
    # Print top 5 and bottom 5 weights
    print(weights.topk(5))
    print(weights.topk(5, largest=False))


if __name__ == "__main__":
    main()
