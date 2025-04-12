import autorootcwd
import json
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


def sanitize_json_filenames(json_path, output_path=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_data = {}
    for split in ["train", "val", "test"]:
        if split in data:
            new_data[split] = [sanitize_filename(fname) for fname in data[split]]

    # Default to overwriting if no output path is given
    output_path = output_path or json_path

    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Saved sanitized JSON to: {output_path}")


# 🔧 Usage
sanitize_json_filenames("./data/processed/splits.json")
