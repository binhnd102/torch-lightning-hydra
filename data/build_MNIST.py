import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split



class ImageDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def build_MNIST(data_dir, valid_size=0.3):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Set transformation
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # load the dataset
    # load the same training set but different transformation
    mnist_train = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    data, targets = mnist_train.data, mnist_train.targets

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=valid_size)

    mnist_train = ImageDataset(X_train, y_train, transform=train_transform)

    mnist_val = ImageDataset(X_test, y_test, transform=valid_transform)

    mnist_test = datasets.MNIST(root=data_dir, train=False, transform=valid_transform)
    return mnist_train, mnist_val, mnist_test
