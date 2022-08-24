"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS,
    sample_size = None: float
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  if sample_size:
    train_targets = train_data.targets
    test_targets = test_data.targets

    from sklearn.model_selection import train_test_split
    import numpy as np 
    train_subset_1_idx, train_subset_2_idx = train_test_split(np.arange(len(train_targets)),train_size=sample_size,shuffle=True,stratify=train_targets)
    test_subset_1_idx, test_subset_2_idx = train_test_split(np.arange(len(test_targets)),train_size=sample_size,shuffle=True,stratify=test_targets)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_subset_1_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_subset_1_idx)

  # Get class names and class to idx
  class_names = train_data.classes
  class_to_idx = train_data.class_to_idx

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      sampler=train_sampler,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      sampler=test_sampler,      
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names, class_to_idx
