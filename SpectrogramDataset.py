from pathlib import Path

from torch import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, spec_fold_dirs, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.spec_paths = []
        for fold_dir in spec_fold_dirs:
            self.spec_paths.extend(list(Path(fold_dir).glob("*.spec")))

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, idx):
        file_path = self.spec_paths[idx]
        spec = torch.load(file_path)
        if self.transform:
            spec = self.transform(spec)

        file_name = Path(file_path).name
        parts = file_name.split("-")
        label = int(parts[1])
        if self.target_transform:
            label = self.target_transform(label)

        pos = file_name.find(".wav")
        original_filename = file_name[: pos + 4]

        return spec, label, original_filename
