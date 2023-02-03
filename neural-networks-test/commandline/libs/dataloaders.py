from torch.utils.data import DataLoader
from torchvision import datasets as Datasets

from libs.transforms import data_transforms

def get_dataloaders(data_directory, batch_size):
  image_datasets = {
    "train": Datasets.ImageFolder(root=f"{data_directory}/train", transform=data_transforms["train"]),
    "validate": Datasets.ImageFolder(root=f"{data_directory}/valid", transform=data_transforms["validate"]),
    "test": Datasets.ImageFolder(root=f"{data_directory}/test", transform=data_transforms["validate"]),
  }

  dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size = batch_size, shuffle=True),
    "validate": DataLoader(image_datasets["validate"], batch_size = batch_size, shuffle=True),
    "test": DataLoader(image_datasets["test"], batch_size = batch_size, shuffle=True)
  }

  return dataloaders, image_datasets
