from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

DATA_PATH = 'data'

def get_y_fn(x):
    return Path(str(x.parent)+'mask')/x.name

# one class
codes = array(['tumor'])
# normalize, this is x/255, mean is 0, std is 1. It is according to your data 
custom_stats = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

src_size=np.array([512,512])
bs,size = 8,src_size//2

src = (SegmentationItemList.from_folder(DATA_PATH).split_by_folder(valid='val').label_from_func(get_y_fn,classes=codes))
data = (src.transform(get_transforms(),size=size,tfm_y=True).databunch(bs=bs).normalize(custom_stats))

# dice for one class
def dice(input, target):
    input = torch.sigmoid(input)
    inputs = (input > 0.5).float()
    targets = target.float()
    return 2. * (inputs * targets).sum() / (inputs.sum() + targets.sum())

metrics=dice
wd=1e-2

def criterion(input,target):
    return F.binary_cross_entropy_with_logits(input, target.float())

learn = unet_learner(data, models.resnet34, loss_func=criterion, metrics=metrics, wd=wd)

lr=1e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

learn.save('stage-1')

learn.load('stage-1')
learn.unfreeze()
lrs = slice(lr/100,lr)
learn.fit_one_cycle(12, lrs, pct_start=0.8)
learn.save('stage-2')

learn=None
gc.collect()

size = src_size
bs=8

data = (src.transform(get_transforms(),size=size,tfm_y=True).databunch(bs=bs).normalize(custom_stats))
learn = unet_learner(data, models.resnet34, loss_func=criterion, metrics=metrics, wd=wd).load('stage-2')

lr=7e-5
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

learn.save('stage-1-big')
learn.load('stage-1-big')

learn.unfreeze()
lrs = slice(lr/1000,lr/10)
learn.fit_one_cycle(10, lrs)
learn.save('stage-2-big')