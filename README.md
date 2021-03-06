## FLDMN: Feature and Label Distributions Matching Networks for Cross-domain Object Detection



This repository is the official PyTorch implementation of paper [FLDMN: Feature and Label Distributions Matching Networks for Cross-domain
Object Detection](). (The work has been submitted to [AAAI2021](https://aaai.org/Conferences/AAAI-21/aaai21call/))

## Main requirements

  * **torch == 1.0.0**
  * **torchvision == 0.2.0**
  * **Python 3**

## Environmental settings
This repository is developed using python **3.7.6** on Ubuntu **16.04.5 LTS**. The CUDA nad CUDNN version is **9.0** and **7.4.1** respectively. We use **one NVIDIA 1080ti GPU card** for training and testing. Other platforms or GPU cards are not fully tested.

Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) respository to setup the environment.

## Datasets
### Datasets Preparation
* **Cityscape and FoggyCityscape:** Download the [Cityscape](https://www.cityscapes-dataset.com/) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).
* **PASCAL_VOC 07+12:** Please follow the [instruction](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC dataset.
* **Clipart:** Please follow the [instruction](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) to prepare Clipart dataset.
* **Sim10k:** Download the dataset from this [website](https://fcav.engin.umich.edu/sim-dataset/).  

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  

### Data Augmentation
We use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate the transferred samples for both source and target domains, and then train the model with original and transferred images.

### Data Download
**Due to the size limitation of uploaded data, we will release the data before and after augmentation upon acceptance.**

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

**Due to the size limitation of uploaded data, we will release the pre-trained FLDMN models upon acceptance.**

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python da_trainval_net.py \
       --dataset source_dataset/target_dataset \
       --net vgg16/resnet101 \
       --pretrained_path path_to_pre-trained-model \
       --save_dir path_to_save-model
```
## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python test_net.py \
       --dataset source_dataset/target_dataset \
       --net vgg16/resnet101  \
       --model_prefix path_to_model


