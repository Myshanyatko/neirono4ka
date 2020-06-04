from fastai.vision import *

bs = 64
path = Path('train')
path.ls()

#np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=path,valid_pct=0.2, size=224, num_workers=4, bs=bs)
data.normalize(imagenet_stats)

print(data.classes)
data.show_batch(rows=3, figsize=(7,8))