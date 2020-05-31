import torch
import pytorch_lightning as pl
from data import build_dataset
from torch.utils.data import DataLoader
from argparse import Namespace
from sklearn.metrics import accuracy_score
from collections import OrderedDict


class MNISTExperiment(pl.LightningModule):
    """ Simple classifier using Pytorch Lightning"""
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.hparams = Namespace(**hparams)
        self.ds_train, self.ds_val, self.ds_test = build_dataset(self.hparams.dataset,
                                                                 self.hparams.data_dir,
                                                                 self.hparams.validation_split)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.model.loss_func(self(x), y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(y.view_as(pred)).sum() / (x.shape[0] * 1.0)
        return {"batch_val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        accuracy = torch.stack([x['batch_val_acc'].float() for x in outputs]).mean()
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {"log": {"val_acc": accuracy}}

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
        return DataLoader(self.ds_train,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.test_batch_size)