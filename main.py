import pytorch_lightning as pl
import hydra
from experiment import MNISTExperiment
from pytorch_lightning.callbacks import ModelCheckpoint
import os
# DEFAULTS used by the Trainer


@hydra.main(config_path="config.yaml")
def main(cfg):
    # train!
    checkpoint_callback = ModelCheckpoint(
        filepath=cfg.experiment.output_dir,
        save_top_k=True,
        verbose=True,
        monitor='val_acc',
        mode='max',
        prefix=''
    )

    if cfg.experiment.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.experiment.device_id


    model = hydra.utils.instantiate(cfg.model)
    experiment = MNISTExperiment(model=model,
                                 hparams=cfg.experiment)
    num_gpu = 0 if cfg.experiment.device == 'cpu' else len(cfg.experiment.device_id)
    print(cfg.pretty())
    runner = pl.Trainer(max_epochs=cfg.experiment.max_epoch,
                        checkpoint_callback=checkpoint_callback,
                        gpus=num_gpu)
    runner.fit(experiment)
    runner.test()


if __name__ == "__main__":
    main()