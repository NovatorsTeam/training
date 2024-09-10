import torch
from torch.utils.data import DataLoader
from src.hackaton_model_training.data.dataset import RansomSideDataset
from random import randint
import torchvision
import os
from dotenv import load_dotenv

load_dotenv()

BOTTOM_REPLACE_IMAGES = int(os.getenv("TEST_DATASET_BOTTOM_REPLACE_IMAGES"))
BOTTOM_REMAIN_IMAGES = int(os.getenv("TEST_DATASET_BOTTOM_REMAIN_IMAGES"))
SIDE_REPLACE_IMAGES = int(os.getenv("TEST_DATASET_SIDE_REPLACE_IMAGES"))
SIDE_REMAIN_IMAGES = int(os.getenv("TEST_DATASET_SIDE_REMAIN_IMAGES"))


def test_dataset_init(test_dataset_path):
    dataset = RansomSideDataset(test_dataset_path)
    assert len(dataset._botton_files) == BOTTOM_REMAIN_IMAGES + BOTTOM_REPLACE_IMAGES
    assert len(dataset._side_files) == SIDE_REMAIN_IMAGES + SIDE_REPLACE_IMAGES

def test_dataset_len(test_dataset_path):
    dataset = RansomSideDataset(test_dataset_path)
    assert len(dataset) == BOTTOM_REMAIN_IMAGES + BOTTOM_REPLACE_IMAGES

def test_dataset_getitem(test_dataset_path):
    dataset = RansomSideDataset(test_dataset_path)
    idx = 0
    bottom_image, side_image, label = dataset[idx]
    assert isinstance(bottom_image, torch.Tensor)
    assert isinstance(side_image, torch.Tensor)
    assert isinstance(label, bool)

def test_dataset_labels(test_dataset_path):
    dataset = RansomSideDataset(test_dataset_path)
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        assert label in [True, False]

def test_dataloader(test_dataset_path):
    dataset = RansomSideDataset(test_dataset_path, transform=torchvision.transforms.Resize(size=(320, 320)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        assert len(batch) == 3
        assert batch[0].shape[0] == 32
        assert batch[1].shape[0] == 32
        assert batch[2].shape[0] == 32
        break # Enough to test only one batch
        
def test_dataset_transform(test_dataset_path):
    height, width = randint(1, 2048), randint(1, 2048)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(height, width)),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    
    dataset = RansomSideDataset(test_dataset_path, transform=transforms)
    
    bottom_image, side_image, label = dataset[0]
    
    assert bottom_image.shape[-2:] == (height, width)
    assert side_image.shape[-2:] == (height, width)
    
    assert bottom_image.shape[0] == 1
    assert side_image.shape[0] == 1
    
    assert label in [True, False]