import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def load_train_validation_data(
        data_dir: str,
        validation_size=0.1,
        random_seed=42,
        batch_size=8,
        shuffle=True,
):
    # values form Cifar 10 — we could calculate them, but it's easier to just use them 
    normalize = transforms.Normalize(
        mean=[0.5],
        std=[0.5],
    )

    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_split, validation_split = indices[split:], indices[:split]

    train_dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_split),
    )

    validation_dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(validation_split),
    )

    return train_dataset, validation_dataset

def load_test_data(
        data_dir: str,
        shuffle=True,
        batch_size=8,
):
    normalize = transforms.Normalize(
        mean=[0.5],
        std=[0.5],
    )

    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    test_dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return test_dataset
