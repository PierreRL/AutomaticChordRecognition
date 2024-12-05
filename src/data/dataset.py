import autorootcwd
import os
from torch.utils.data import Dataset, random_split
from src.utils import (
    get_cqt,
    chord_ann_to_tensor,
    pitch_shift_cqt,
    transpose_chord_id_vector,
)
import torch
from torch import Tensor


# Create a torch dataset
class FullChordDataset(Dataset):
    def __init__(self, filenames=None, hop_length=4096, cached=True):
        """
        Initialize a chord dataset. Each sample is a tuple of features and chord annotation.
        Args:
            filenames (list) = None: A list of filenames to include in the dataset. If None, all files in the processed audio directory are included.
            hop_length (int): The hop length used to compute the log CQT.
            cached (bool): If True, the dataset loads cached CQT and chord annotation files. If False, the CQT and chord annotation are computed on the fly.
        """
        if not filenames:
            filenames = os.listdir("./data/processed/audio")
            # Filter out non-mp3 files and remove the extension
            filenames = [
                filename.split(".")[0]
                for filename in filenames
                if filename.endswith(".mp3")
            ]

        self.filenames = filenames
        self.hop_length = hop_length
        self.cached = cached
        self.sr = 44100
        self.bins_per_octave = 36
        self.n_bins = self.bins_per_octave * 6
        self.cqt_cache_dir = "./data/processed/cache/cqts"
        self.chord_cache_dir = "./data/processed/cache/chords"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        if self.cached:
            return self.__get_cached_item(idx)
        filename = self.filenames[idx]

        # Load the log CQT and chord annotation
        cqt = get_cqt(
            filename,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )
        chord_ids = chord_ann_to_tensor(
            filename, frame_length=self.hop_length / self.sr
        )

        return self.get_minimum_length_frame(cqt, chord_ids)

    def get_minimum_length_frame(
        self, cqt: Tensor, chord_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns the cqt and chord annotation with the minimum length.

        Args:
            cqt (Tensor): The log CQT.
            chord_one_hot (Tensor): The chord annotation.

        Returns:
            cqt (Tensor): The log CQT with the minimum length.
            chord_one_hot (Tensor): The chord annotation with the minimum length.
        """
        minimum_length_frames = min(cqt.shape[0], chord_ids.shape[0])
        return cqt[:minimum_length_frames], chord_ids[:minimum_length_frames]

    def __get_cached_item(self, idx):
        filename = self.filenames[idx]
        cqt = torch.load(f"{self.cqt_cache_dir}/{filename}.pt", weights_only=True)
        chord_one_hot = torch.load(
            f"{self.chord_cache_dir}/{filename}.pt", weights_only=True
        )
        return self.get_minimum_length_frame(cqt, chord_one_hot)

    def get_filename(self, idx):
        return self.filenames[idx]


class FixedLengthRandomChordDataset(Dataset):
    """
    A chord dataset that returns fixed length frames, where the frame location in each sample of audio is uniformly random.
    """

    def __init__(
        self,
        segment_length=10,
        hop_length=4096,
        cached=True,
        random_pitch_shift=True,
        filenames=None,
    ):
        """
        Initialize a chord dataset. Each sample is a tuple of features and chord annotation.
        Args:
            frame_length (int): The length of the frame in seconds.
            hop_length (int): The hop length used to compute the log CQT.
            cached (bool): If True, the dataset loads cached CQT and chord annotation files. If False, the CQT and chord annotation are computed on the fly.
            random_pitch_shift (bool): If True, the dataset randomly shifts the pitch of the cqt.
            filenames (list) = None: A list of filenames to include in the dataset. If None, all files in the processed audio directory are included
        """
        super().__init__()
        self.full_dataset = FullChordDataset(
            hop_length=hop_length, cached=cached, filenames=filenames
        )
        self.segment_length = segment_length
        self.random_pitch_shift = random_pitch_shift

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        full_cqt, full_chord_ids = self.full_dataset[idx]

        # Convert segment length in seconds to segment length in frames
        segment_length_samples = int(
            self.segment_length * self.full_dataset.sr / self.full_dataset.hop_length
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
            cqt_patch = pitch_shift_cqt(
                cqt_patch, semitones, self.full_dataset.bins_per_octave
            )
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
        segment_length=10,
        hop_length=4096,
        cached=True,
        filenames=None,
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
            filenames=filenames, hop_length=hop_length, cached=cached
        )
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.data = self.generate_fixed_segments()

    def generate_fixed_segments(self) -> list[tuple[Tensor, Tensor]]:
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
            self.segment_length * self.full_dataset.sr / self.hop_length
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

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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


if __name__ == "__main__":
    main()
