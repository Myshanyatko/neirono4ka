from fastai.vision import *
from fastai.callbacks import *

bs = 20
path = Path('2 heroes')
path.ls()

#np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=path,valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4, bs=bs)
data.normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,8))
data.c, len(data.train_ds), len(data.valid_ds)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(10)
