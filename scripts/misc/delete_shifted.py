#!/usr/bin/env python3
"""
Script for deleting all files ending in _shifted_{some integer}.mp3 in a given folder.
"""

import os
import re
import argparse

def delete_shifted_files(folder: str):
    """
    Delete files ending with _shifted_{integer}.mp3 in the given folder.

    Parameters:
      folder (str): The directory in which to search and delete matching files.
    """
    # Compile a regex to match filenames ending in _shifted_{integer}.mp3
    pattern = re.compile(r'_shifted_-?\d+\.mp3$')
    
    # List all files in the given folder
    for filename in os.listdir(folder):
        if pattern.search(filename):
            file_path = os.path.join(folder, filename)
            print(f"Deleting: {file_path}")
            os.remove(file_path)

def main():
    parser = argparse.ArgumentParser(
        description="Delete all files ending in _shifted_{integer}.mp3 in a given folder."
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="The folder where the files should be deleted."
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.")
        return

    delete_shifted_files(args.folder)

if __name__ == "__main__":
    main()