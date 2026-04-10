import os
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class GravexDataset(Dataset):
    """Base GRAVEX-200K dataset — yields (PIL Image, label, path)."""

    def __init__(self, manifest_csv: str, data_root: str = None):
        self.samples = []
        with open(manifest_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                path, label = row[0], int(row[1])
                if data_root and not os.path.isabs(path):
                    path = os.path.join(data_root, path)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        return image, label, path
