import autorootcwd
import os
from torch.utils.data import Dataset, random_split
from src.utils import get_cqt, chord_ann_to_tensor
import torch
from torch import Tensor


# Create a torch dataset
class ChordDataset(Dataset):
    def __init__(self, filenames=None, hop_length=2048, cached=True):
        """
        Initialize a chord dataset. Each sample is a tuple of features and chord annotation.
        Args:
            filenames (list) = None: A list of filenames to include in the dataset. If None, all files in the processed audio directory are included.
            hop_length (int): The hop length used to compute the log CQT.
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
        self.sr = 22050
        self.cqt_cache_dir = "./data/processed/cache/cqts"
        self.chord_cache_dir = "./data/processed/cache/chords"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        if self.cached:
            return self.__get_cached_item(idx)
        filename = self.filenames[idx]

        # Load the log CQT and chord annotation
        cqt = get_cqt(filename, hop_length=self.hop_length)
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


def main():
    import torch

    torch.manual_seed(0)
    filenames = os.listdir("./data/processed/audio")

    # Filter out non-mp3 files and remove the extension
    filenames = [
        filename.split(".")[0] for filename in filenames if filename.endswith(".mp3")
    ]

    # Create a dataset
    dataset = ChordDataset(filenames)

    # Split the dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Number of training samples: {len(train_dataset)}")


if __name__ == "__main__":
    main()
