import torch
import pytorch_lightning as pl


class MNISTExperiment(pl.LightningModule):
    """ Simple classifier using Pytorch Lightning"""
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.model.loss_func(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params.lr)