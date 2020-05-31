import torch
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import Namespace
from hydra import utils
import os


class MNISTExperiment(pl.LightningModule):
    """ Simple classifier using Pytorch Lightning"""
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.hparams = Namespace(**hparams)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.model.loss_func(self(x), y)
        return {'loss': loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(y.view_as(pred)).sum() / (x.shape[0] * 1.0)
        return {"batch_test_acc": accuracy}

    def test_epoch_end(self, outputs):
        accuracy = torch.stack([x['batch_test_acc'].float() for x in outputs]).mean()
        return {"log": {"test_acc": accuracy}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        wording_dir = utils.get_original_cwd()
        root_dir = os.path.join(wording_dir, self.hparams.data_dir)
        print(root_dir)
        mnist_train = MNIST(
            root=root_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        return DataLoader(mnist_train,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          shuffle=True)


    def test_dataloader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        wording_dir = utils.get_original_cwd()
        root_dir = os.path.join(wording_dir, self.hparams.data_dir)
        mnist_test = MNIST(
            root=root_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        return DataLoader(mnist_test,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.test_batch_size)