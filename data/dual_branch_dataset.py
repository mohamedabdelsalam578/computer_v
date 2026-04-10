import os
import csv
import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class DualBranchDataset(Dataset):
    """
    Dataset that yields (face_crop_tensor, full_image_tensor, label)
    using pre-computed face crops from the manifest CSV.
    """

    def __init__(
        self,
        manifest_csv: str,
        face_transform=None,
        full_transform=None,
        face_size: int = 256,
        random_face_select: bool = True,
        path_prefix_override: str = None,   # RunPod: replace Mac prefix with pod path
        original_prefix: str = None,
    ):
        self.face_transform = face_transform
        self.full_transform = full_transform
        self.face_size = face_size
        self.random_face_select = random_face_select
        self.samples = []

        def _remap(p: str) -> str:
            """Replace Mac absolute prefix with RunPod prefix if set."""
            if path_prefix_override and original_prefix and p.startswith(original_prefix):
                return path_prefix_override + p[len(original_prefix):]
            return p

        with open(manifest_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'image_path': _remap(row['image_path']),
                    'label': int(row['label']),
                    'num_faces': int(row['num_faces']),
                    'face_crops': [_remap(p) for p in json.loads(row['face_crop_paths'])] if row['face_crop_paths'] else [],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        label = sample['label']
        face_crop_paths = sample['face_crops']

        # Load full image and apply transforms
        full_image = Image.open(image_path).convert('RGB')
        if self.full_transform:
            full_tensor = self.full_transform(full_image)
        else:
            full_tensor = torch.zeros(3, self.face_size, self.face_size)

        # Load face crop
        if face_crop_paths:
            if self.random_face_select:
                crop_path = random.choice(face_crop_paths)
            else:
                crop_path = face_crop_paths[0]
            face_image = Image.open(crop_path).convert('RGB')
            if self.face_transform:
                face_tensor = self.face_transform(face_image)
            else:
                face_tensor = torch.zeros(3, self.face_size, self.face_size)
        else:
            # No face detected — use black placeholder
            face_tensor = torch.zeros(3, self.face_size, self.face_size)

        return face_tensor, full_tensor, label
