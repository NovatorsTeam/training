import os
import shutil
from random import randint

import pytest
import torch
from dotenv import load_dotenv
from torchvision.utils import save_image

load_dotenv()

BOTTOM_REPLACE_IMAGES = int(os.getenv("TEST_DATASET_BOTTOM_REPLACE_IMAGES"))
BOTTOM_REMAIN_IMAGES = int(os.getenv("TEST_DATASET_BOTTOM_REMAIN_IMAGES"))
SIDE_REPLACE_IMAGES = int(os.getenv("TEST_DATASET_SIDE_REPLACE_IMAGES"))
SIDE_REMAIN_IMAGES = int(os.getenv("TEST_DATASET_SIDE_REMAIN_IMAGES"))


def create_random_image() -> torch.Tensor:
    return torch.rand([3, randint(4, 2048), randint(4, 2048)])


@pytest.fixture(scope="session")
def test_dataset_path():
    test_dir_path = os.path.join("data", "Test", "test_data")
    if not os.path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    folders_pairs = [["bottom", "replacment"], ["bottom", "remain"], ["side", "replacment"], ["side", "remain"]]

    for pair in folders_pairs:
        if not os.path.exists(os.path.join(test_dir_path, pair[0], pair[1])):
            os.makedirs(os.path.join(test_dir_path, pair[0], pair[1]))

    for folders, n_images in zip(
        folders_pairs, [BOTTOM_REPLACE_IMAGES, BOTTOM_REMAIN_IMAGES, SIDE_REPLACE_IMAGES, SIDE_REMAIN_IMAGES]
    ):
        for i in range(n_images):
            image = create_random_image()
            save_image(image, os.path.join(test_dir_path, folders[0], folders[1], f"image_{i}.jpg"))

    yield test_dir_path

    shutil.rmtree(test_dir_path)
