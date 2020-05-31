import pytorch_lightning as pl
import hydra
from experiment import MNISTExperiment

@hydra.main(config_path="config.yaml")
def main(cfg):
    # train!
    model = hydra.utils.instantiate(cfg.model)
    experiment = MNISTExperiment(model=model,
                                 hparams=cfg.experiment)
    runner = pl.Trainer(max_epochs=cfg.experiment.max_epoch)
    runner.fit(experiment)


if __name__ == "__main__":
    main()