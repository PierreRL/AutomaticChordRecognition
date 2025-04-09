"""
Filter and rename the audio and chord files to match the metadata in the JAMS files.

Copies data from the raw directory to the processed directory. During the copy, files are renamed to the format artist_title where artist and title are available in the JAMS file. Otherwise, the filename is kept the same.

Filters the audio and chord files to only keep the first instance of each new_filename.

This script is run once to prepare the data for training.
"""

import autorootcwd
import jams
import os
import shutil
from tqdm import tqdm
from src.utils import get_annotation_metadata


def main():
    with open("./data/raw/audio/filelist.txt", "r") as f:
        filenames = f.read().splitlines()

    # Just get the root filename
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    # Construct dict for each file
    metadata = [get_relevant_metadata(f) for f in filenames]

    # Keep only the first instance of each new_filename
    metadata = {m["new_filename"]: m for m in metadata}

    # Create new dict of old_filename -> new_filename
    filename_map = {m["filename"]: m["new_filename"] for m in metadata.values()}

    errored_files = rename_copy_files(
        filename_map, "./data/raw/audio", "./data/processed/audio", "mp3"
    )
    errored_chords = rename_copy_files(
        filename_map, "./data/raw/references_v2", "./data/processed/chords", "jams"
    )
    len(errored_files), len(errored_chords)


def get_relevant_metadata(filename: str) -> dict:
    """
    Get the metadata from a JAMS file revelant to renaming the file.

    Args:
        filename (str): The filename of the JAMS file to load.

    Returns:
        metadata (dict): The metadata of the JAMS file.
    """
    metadata = get_annotation_metadata(filename)
    artist = metadata.artist
    title = metadata.title
    # Strip, remove spaces and slashes and lowercase
    artist, title = artist.lower(), title.lower()
    artist, title = artist.strip(), title.strip()
    artist, title = artist.replace(" ", ""), title.replace(" ", "")
    artist, title = artist.replace("/", ""), title.replace("/", "")
    artist, title = artist.replace(".", ""), title.replace(".", "")

    duration = metadata.duration

    new_filename = f"{artist}_{title}"
    if artist == "" and title == "":
        new_filename = filename

    return {
        "new_filename": new_filename,
        "artist": artist,
        "title": title,
        "filename": filename,
        "duration": duration,
    }


def rename_copy_files(
    filename_map: dict, old_dir: str, new_dir: str, extension: str
) -> list:
    """
    Copy and rename files from old_dir to new_dir using filenames in filename_map.

    Args:
        filename_map (dict): A dict mapping old filenames to new filenames.
        old_dir (str): The directory containing the old files.
        new_dir (str): The directory to copy the files to.
        extension (str): The file extension to use.

    Returns:
        errored_files (list): A list of filenames that could not be copied.
    """
    errored_files = []
    for old_filename, new_filename in tqdm(filename_map.items()):
        try:
            shutil.copy(
                f"{old_dir}/{old_filename}.{extension}",
                f"{new_dir}/{new_filename}.{extension}",
            )
        except Exception as e:
            print(e)
            errored_files.append(old_filename)

    return errored_files


if __name__ == "__main__":
    main()
