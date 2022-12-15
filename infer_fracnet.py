from time import perf_counter

import nibabel
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import yaml

from jawfrac.datamodules import FracNetDataModule
from jawfrac.models import FracNet


def infer(regex_filter: str=''):
    with open('jawfrac/config/fracnet.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if regex_filter:
            config['datamodule']['root'] = config['datamodule']['root'] + regex_filter
            config['datamodule']['regex_filter'] = ''

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].pop('batch_size')
    dm = FracNetDataModule(
        linear=True,
        displacements=True,
        seed=config['seed'],
        batch_size=1,
        **config['datamodule'],
    )

    model = FracNet.load_from_checkpoint(
        'checkpoints/fracnet.ckpt',
        num_classes=dm.num_classes,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.test(model, datamodule=dm)

    for i, pred in enumerate(tqdm(preds)):
        path = dm.root / dm.predict_dataset.files[i][0]

        img = nibabel.load(path)
        file = path.parent / 'frac_pred_fracnet.nii.gz'
        img = nibabel.Nifti1Image(pred.cpu().numpy().astype(np.uint16), img.affine)
        nibabel.save(img, file)


if __name__ == '__main__':
    infer()


    times = []
    for regex_filter in [
        # 'Annotation UK/1/', 'Annotation UK/109/', 'Annotation UK/11/', 'Annotation UK/114/', 'Annotation UK/119/', 'Annotation UK/121/', 'Annotation UK/122/', 'Annotation UK/125/', 'Annotation UK/126/', 'Annotation UK/127/', 'Annotation UK/132/', 'Annotation UK/134/', 'Annotation UK/141/', 'Annotation UK/149/', 'Annotation UK/150/', 'Annotation UK/157/', 'Annotation UK/159/', 'Annotation UK/17/', 'Annotation UK/173/', 'Annotation UK/182/', 'Annotation UK/186/', 'Annotation UK/188/', 'Annotation UK/189/', 'Annotation UK/192/', 'Annotation UK/194/', 'Annotation UK/25/', 'Annotation UK/31/', 'Annotation UK/34/', 'Annotation UK/35/', 'Annotation UK/36/', 'Annotation UK/42/', 'Annotation UK/45/', 'Annotation UK/48/', 'Annotation UK/55/', 'Annotation UK/67/',
        'Controls/Patient 56-60/DICOM/Patient 59/', 'Controls/Patient 61-65/DICOM/Patient 64/', 'Controls/Patient 28-40/DICOM/Patient 36/', 'Controls/Patient 51-55/DICOM/Patient 51/', 'Controls/Patient 41-45/DICOM/Patient 42/', 'Controls/Patient 4/', 'Controls/Patient 76-80/DICOM/Patient 76/', 'Controls/Patient 56-60/DICOM/Patient 60/', 'Controls/Patient 46-50/DICOM/Patient 48/', 'Controls/Patient 16-21/DICOM/Patient 19/', 'Controls/Patient 5-10/DICOM/Patient 9/', 'Controls/Patient 66-70/DICOM/Patient 66/', 'Controls/Patient 5-10/DICOM/Patient 7/', 'Controls/Patient 71-75/DICOM/Patient 71/', 'Controls/Patient 46-50/DICOM/Patient 47/', 'Controls/Patient 76-80/DICOM/Patient 77/', 'Controls/Patient 41-45/DICOM/Patient 41/', 'Controls/Patient 56-60/DICOM/Patient 57/', 'Controls/Patient 66-70/DICOM/Patient 67/', 'Controls/Patient 61-65/DICOM/Patient 63/', 'Controls/Patient 16-21/DICOM/Patient 17/', 'Controls/Patient 56-60/DICOM/Patient 58/', 'Controls/Patient 76-80/DICOM/Patient 80/', 'Controls/Patient 11-15/DICOM/Patient 15/', 'Controls/Patient 61-65/DICOM/Patient 65/', 'Controls/Patient 66-70/DICOM/Patient 69/', 'Controls/Patient 11-15/DICOM/Patient 13/', 'Controls/Patient 81-85/DICOM/Patient 84/', 'Controls/Patient 51-55/DICOM/Patient 53/', 'Controls/Patient 23-27/DICOM/Patient 23/', 'Controls/Patient 41-45/DICOM/Patient 44/', 'Controls/Patient 11-15/DICOM/Patient 14/', 'Controls/Patient 66-70/DICOM/Patient 70/', 'Controls/Patient 22/DICOM/STD00001/', 'Controls/Patient 41-45/DICOM/Patient 45/',
    ]:
        t = perf_counter()

        infer(regex_filter=regex_filter)
        times.append(perf_counter() - t)
        print(f'Time: {perf_counter() - t}.')

    times = np.array(times)
    print(f'Total times: {times.mean()} +- {times.std()}')
