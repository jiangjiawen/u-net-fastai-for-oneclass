# u-net-fastai-for-oneclass
new version fastai unet for one class, the example in the doc is for multi classes

according to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb

if you want to use multi gpu, you can use this method:

https://github.com/fastai/fastai/blob/5a04f96bcff36a53847ff0b93648db16990c81fe/docs/distributed.md

If you have multi machine with multi GPU, you can according to this:

https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

unet_learner can not use DataParallel, according to this https://github.com/fastai/fastai/issues/1435 .
