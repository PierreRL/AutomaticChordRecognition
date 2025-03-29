import autorootcwd
import os
from torch.utils.data import Dataset, random_split
import torch
from torch import Tensor
from typing import Tuple, List

from src.utils import (
    pitch_shift_cqt,
    transpose_chord_id_vector,
    SMALL_VOCABULARY,
    NUM_CHORDS,
    HOP_LENGTH,
    SR,
    BINS_PER_OCTAVE,
)


# Create a torch dataset
class FullChordDataset(Dataset):
    def __init__(
        self,
        filenames: List[str] = None,
        hop_length: int = HOP_LENGTH,
        generative_features: bool = False,
        mask_X: bool = False,
        input_dir: str = "./data/processed/",
        small_vocab: bool = SMALL_VOCABULARY,
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
        """
        if not filenames:
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
        self.generative_features = generative_features
        self.mask_X = mask_X
        self.cqt_cache_dir = f"{self.input_dir}/cache/{self.hop_length}/cqts"
        if self.generative_features:
            self.gen_cache_dir = f"{self.input_dir}/cache/{self.hop_length}/gen"
        self.chord_cache_dir = f"{self.input_dir}/cache/{self.hop_length}/chords"
        self.small_vocab = small_vocab
        if self.small_vocab:
            self.chord_cache_dir = (
                f"{self.input_dir}/cache/{self.hop_length}/chords_small_vocab"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if self.generative_features:
            cqt, chord_ids, gen = self.__get_cached_item(idx)
        else:
            cqt, chord_ids = self.__get_cached_item(idx)

        if self.mask_X:
            chord_ids = torch.where(chord_ids == 1, -1, chord_ids)

        if self.generative_features:
            return self.get_minimum_length_frame(cqt, chord_ids, gen)
        else:
            return self.get_minimum_length_frame(cqt, chord_ids)
    
    def get_minimum_length_frame(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        """
        Returns each tensor trimmed to the minimum length along the first dimension
        among all provided tensors.

        Args:
            *tensors (Tensor): An arbitrary number of tensors.

        Returns:
            Tuple[Tensor, ...]: A tuple of tensors, each sliced to the minimum length.
        """
        if not tensors:
            raise ValueError("At least one tensor must be provided")

        # Compute the minimum length among all tensors along the first dimension.
        min_length = min(tensor.shape[0] for tensor in tensors)
        return tuple(tensor[:min_length] for tensor in tensors)

    def get_class_weights(self, epsilon=1e1, alpha=0.65) -> torch.Tensor:
        """
        Calculate the chord loss weights for the dataset.

        Args:
            epsilon (float): A value to prevent division by zero. The effective number of samples for each class is epsilon + counts.
            alpha (float): The exponent of the inverse frequency. The higher the value, the more the weights are skewed towards the rare classes. Should be in [0,1].

        Returns:
            weights (Tensor): The chord loss weights, normalized.
        """
        all_chord_ids = torch.cat(
            [self[i][1].flatten() for i in range(len(self))]
        )  # Flatten all chord IDs
        all_chord_ids = all_chord_ids[all_chord_ids != -1]  # Remove -1 (ignored labels)

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

    def __get_cached_item(self, idx):
        filename = self.filenames[idx]
        cqt = torch.load(f"{self.cqt_cache_dir}/{filename}.pt", weights_only=True)
        chord_ids = torch.load(
            f"{self.chord_cache_dir}/{filename}.pt", weights_only=True
        )
        if self.generative_features:
            gen = torch.load(f"{self.gen_cache_dir}/{filename}.pt", weights_only=True)
            return cqt, chord_ids, gen
        else:
            return cqt, chord_ids

    def get_filename(self, idx):
        return self.filenames[idx]


class FixedLengthRandomChordDataset(Dataset):
    """
    A chord dataset that returns fixed length frames, where the frame location in each sample of audio is uniformly random.
    """

    def __init__(
        self,
        random_pitch_shift=False,
        segment_length=10,
        filenames=None,
        hop_length=HOP_LENGTH,
        mask_X=False,
        input_dir="./data/processed/",
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
            input_dir=input_dir,
        )
        self.random_pitch_shift = random_pitch_shift
        self.segment_length = segment_length

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        full_cqt, full_chord_ids = self.full_dataset[idx]

        # Convert segment length in seconds to segment length in frames
        segment_length_samples = int(
            self.segment_length * (SR / self.full_dataset.hop_length)
        )

        if full_cqt.shape[0] > segment_length_samples:
            # If the full data is longer than the desired frame length, take a random slice
            start_idx = torch.randint(
                0, full_cqt.shape[0] - segment_length_samples, (1,)
            ).item()
            cqt_patch = full_cqt[start_idx : start_idx + segment_length_samples]
            chord_ids_patch = full_chord_ids[
                start_idx : start_idx + segment_length_samples
            ]
        else:
            # If the full data is shorter than the desired segment length, pad it
            pad_length = segment_length_samples - full_cqt.shape[0]
            cqt_patch = torch.cat(
                (full_cqt, torch.zeros((pad_length, full_cqt.shape[1])))
            )
            chord_ids_patch = torch.cat(
                (
                    full_chord_ids,
                    torch.full((pad_length,), -1),
                )  # Use -1 as a padding label
            )

        if self.random_pitch_shift:
            semitones = torch.randint(-5, 6, (1,)).item()
            cqt_patch = pitch_shift_cqt(cqt_patch, semitones, BINS_PER_OCTAVE)
            chord_ids_patch = transpose_chord_id_vector(chord_ids_patch, semitones)

        return cqt_patch, chord_ids_patch

    def __len__(self):
        return len(self.full_dataset)


class FixedLengthChordDataset(Dataset):
    """
    A chord dataset that returns fixed length frames, that splits up the audio in a test set into fixed length frames.
    """

    def __init__(
        self,
        segment_length=28,
        filenames=None,
        hop_length=HOP_LENGTH,
        mask_X=False,
        input_dir="./data/processed/",
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

        # Convert segment length in seconds to segment length in frames
        self.segment_length_samples = int(
            self.segment_length * SR / self.full_dataset.hop_length
        )

        #  Loop over each song in the dataset
        for i in range(len(self.full_dataset)):
            cqt, chord_ids = self.full_dataset[i]
            # Loop over each 'segment' in the song
            for j in range(0, cqt.shape[0], self.segment_length_samples):
                data.append(
                    (
                        cqt[j : j + self.segment_length_samples],
                        chord_ids[j : j + self.segment_length_samples],
                    )
                )
        return data

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def generate_datasets(
    train_filenames: List[str],
    val_filenames: List[str],
    test_filenames: List[str],
    input_dir: str,
    segment_length: int,
    mask_X: bool,
    hop_length: int,
    random_pitch_shift: bool = True,
    subset_size=None,
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
    if subset_size:
        train_filenames = train_filenames[:subset_size]
        val_filenames = val_filenames[:subset_size]
        test_filenames = test_filenames[:subset_size]
    train_dataset = FixedLengthRandomChordDataset(
        filenames=train_filenames,
        segment_length=segment_length,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
        random_pitch_shift=random_pitch_shift,
    )
    val_dataset = FixedLengthChordDataset(
        filenames=val_filenames,
        segment_length=segment_length,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
    )
    test_dataset = FullChordDataset(
        filenames=test_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
    )
    train_final_test_dataset = FullChordDataset(
        filenames=train_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
    )
    val_final_test_dataset = FullChordDataset(
        filenames=val_filenames,
        hop_length=hop_length,
        mask_X=mask_X,
        input_dir=input_dir,
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_final_test_dataset,
        val_final_test_dataset,
    )


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
