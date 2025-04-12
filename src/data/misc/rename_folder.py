import autorootcwd
import os
import re
import unicodedata


def sanitize_filename(name):
    # Normalize Unicode (e.g., é vs e + ́)
    name = unicodedata.normalize("NFC", name)

    # Lowercase
    name = name.lower()

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove all characters *except* letters, numbers, underscores, hyphens, and dots
    name = re.sub(r"[^\w\-.]", "", name)

    return name


def sanitize_filenames_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Rename files
        for name in files:
            sanitized = sanitize_filename(name)
            if name != sanitized:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, sanitized)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")

        # Rename directories
        for name in dirs:
            sanitized = sanitize_filename(name)
            if name != sanitized:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, sanitized)
                os.rename(old_path, new_path)
                print(f"Renamed dir:  {old_path} -> {new_path}")


# 🔧 Usage
sanitize_filenames_in_folder("./data/processed/")
