import pytorch_lightning as pl
import yaml

from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule


def infer(checkpoint, interpolation='fast', root=None, channels_list=[16, 32, 64, 128]):
    with open('jawfrac/config/mandibles.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if root is not None:
            config['datamodule']['root'] = root
        config['model']['interpolation'] = interpolation
        config['model']['channels_list'] = channels_list

    batch_size = config['datamodule'].pop('batch_size')
    dm = MandibleSegDataModule(
        seed=config['seed'], batch_size=1, **config['datamodule'],
    )

    model = MandibleSegModule.load_from_checkpoint(
        checkpoint,
        num_classes=dm.num_classes,
        batch_size=batch_size,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)

    return preds


if __name__ == '__main__':
    while True:
        infer('checkpoints/mandibles.ckpt', interpolation='fast')
