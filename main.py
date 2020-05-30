import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
import hydra
from experiment import MNISTExperiment

@hydra.main(config_path="config.yaml")
def main(cfg):
    # train!
    train_loader = hydra.utils.instantiate(cfg.dataloader)
    model = hydra.utils.instantiate(cfg.model)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    experiment = MNISTExperiment(model=model,
                                 params=cfg.experiment)
    runner = pl.Trainer(logger=logger)
    runner.fit(experiment, train_loader)


if __name__ == "__main__":
    main()