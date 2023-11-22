"""
Processes and uploads the UrbanSound8K audio dataset to Hugging Face Hub.

This script reads audio metadata from a CSV file, constructs file paths, and uploads the dataset to Hugging Face for public use. It includes file paths, labels, fold information, and other relevant metadata.

Requires pandas and Hugging Face datasets library. Ensure Hugging Face CLI login before use.
"""

import os
import pandas as pd
from pathlib import Path
from datasets import Audio, Dataset

dataset_name = "danavery/urbansound8k"
source_dir = Path("/Users/davery/urbansound8k")
metadata = pd.read_csv(source_dir / "UrbanSound8K.csv")


def add_path(row):
    fold_path = source_dir / f'fold{row["fold"]}'
    return os.path.join(fold_path, row["slice_file_name"])


metadata["filepath"] = metadata.apply(add_path, axis=1)
dataset_dict = {
    "audio": metadata["filepath"].tolist(),
    "slice_file_name": metadata["slice_file_name"].to_list(),
    "fsID": metadata["fsID"].to_list(),
    "start": metadata["start"].to_list(),
    "end": metadata["end"].to_list(),
    "salience": metadata["salience"].tolist(),
    "fold": metadata["fold"].tolist(),
    "classID": metadata["classID"].tolist(),
    "class": metadata["class"].tolist(),
}

hf_dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio())

print(hf_dataset)
hf_dataset.push_to_hub(dataset_name)
