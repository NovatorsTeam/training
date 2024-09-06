from glob import glob
from os.path import join
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F


class RansomSideDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self._botton_files = glob(join(path, "bottom", "*", "*.jpg"))
        self._side_files = glob(join(path, "side", "*", "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self._botton_files)

    def __getitem__(self, index):
        bottom_image_path = self._botton_files[index]
        side_images_paths = np.random.choice(self._side_files, 4, replace=False)

        is_replacment: bool = Path(bottom_image_path).parent.name == "replacment" or any(
            [Path(path).parent.name == "replacment" for path in side_images_paths]
        )

        bottom_image = torchvision.io.read_image(bottom_image_path)
        height, width = bottom_image.shape[-2:]

        side_images = [F.resize(torchvision.io.read_image(path), (height // 4, width)) for path in side_images_paths]
        side_concatination = F.resize(torch.cat(side_images, dim=1), (height, width))

        assert side_concatination.shape[-2:] == bottom_image.shape[-2:]

        if self.transform:
            bottom_image = self.transform(bottom_image)
            side_concatination = self.transform(side_concatination)

        return bottom_image, side_concatination, is_replacment
