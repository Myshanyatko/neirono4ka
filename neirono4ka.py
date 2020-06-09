from fastai.vision import *
from fastai.callbacks import *

bs = 16
path = Path('heroes2')
path.ls()

#np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=path,valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4, bs=bs)
data.normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,8))
data.c, len(data.train_ds), len(data.valid_ds)

learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(10, callbacks=[EarlyStoppingCallback(learn, patience=2)])
learn.recorder.plot_losses()
learn.recorder.plot()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-3,1e-2))
learn.recorder.plot_losses()
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-8,1e-6))
learn.recorder.plot_losses()


#learn.save('stage-1')
#learn.export()

#img = open_image(path/'0.jpg')
#img
#pred_class,pred_idx,outputs = learn.predict(img)
#print(pred_class)

