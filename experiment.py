import torch
import pytorch_lightning as pl
from data import build_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from collections import OrderedDict


class MNISTExperiment(pl.LightningModule):
    """ Simple classifier using Pytorch Lightning"""
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.params = hparams
        self.ds_train, self.ds_val, self.ds_test = build_dataset(self.params.dataset,
                                                                 self.params.data_dir,
                                                                 self.params.validation_split)

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
        correct = pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct / x.size(0)
        return {"batch_val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x["batch_val_acc"] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {"log": {"val_acc": accuracy}}

    def test_step(self, batch, batch_nb):
        with torch.no_grad():
            x, y = batch
            loss = self.model.loss_func(self(x), y)
            tensorboard_logs = {'test_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params.lr)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          num_workers=self.params.num_workers,
                          batch_size=self.params.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          num_workers=self.params.num_workers,
                          batch_size=self.params.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          num_workers=self.params.num_workers,
                          batch_size=self.params.batch_size)