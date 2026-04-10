import random
import io
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from configs.base_config import FERRETNET_NORMALIZE_MEAN, FERRETNET_NORMALIZE_STD


class JPEGCompression:
    """JPEG compression augmentation via PIL BytesIO round-trip."""

    def __init__(self, quality_range=(65, 100), probability=0.5):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()


class RandomInterpolationResize:
    """Resize with randomly chosen interpolation method."""

    INTERPOLATIONS = [Image.BILINEAR, Image.BICUBIC, Image.NEAREST, Image.LANCZOS]

    def __init__(self, size):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        interp = random.choice(self.INTERPOLATIONS)
        return img.resize(self.size, interp)


class GammaCorrection:
    """Apply gamma correction: img = img ** gamma."""

    def __init__(self, gamma_range=(0.8, 1.2), probability=0.2):
        self.gamma_range = gamma_range
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class RandomBlur:
    """Randomly apply Gaussian, Median, or Bilateral blur."""

    def __init__(self, p_gaussian=0.15, p_median=0.1, p_bilateral=0.1):
        self.p_gaussian = p_gaussian
        self.p_median = p_median
        self.p_bilateral = p_bilateral

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random()
        if r < self.p_gaussian:
            sigma = random.uniform(0.1, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif r < self.p_gaussian + self.p_median:
            img_np = np.array(img)
            ksize = random.choice([3, 5])
            img_np = cv2.medianBlur(img_np, ksize)
            return Image.fromarray(img_np)
        elif r < self.p_gaussian + self.p_median + self.p_bilateral:
            img_np = np.array(img)
            img_np = cv2.bilateralFilter(img_np, d=5, sigmaColor=50, sigmaSpace=50)
            return Image.fromarray(img_np)
        return img


class RandomSharpen:
    """Apply UnsharpMask sharpening."""

    def __init__(self, probability=0.15):
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def get_train_transforms(size=256):
    """Full training augmentation pipeline for both face crops and full images."""
    return T.Compose([
        JPEGCompression(quality_range=(65, 100), probability=0.5),
        RandomInterpolationResize((size, size)),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08),
        GammaCorrection(gamma_range=(0.8, 1.2), probability=0.2),
        RandomBlur(p_gaussian=0.15, p_median=0.1, p_bilateral=0.1),
        RandomSharpen(probability=0.15),
        T.RandomEqualize(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=FERRETNET_NORMALIZE_MEAN, std=FERRETNET_NORMALIZE_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms(size=256):
    """Validation/test transforms — no augmentation."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=FERRETNET_NORMALIZE_MEAN, std=FERRETNET_NORMALIZE_STD),
    ])
