## FLDMN: Feature and Label Distributions Matching Networks for Cross-domain Object Detection



This repository is the official PyTorch implementation of paper [FLDMN: Feature and Label Distributions Matching Networks for Cross-domain
Object Detection](). (The work has been submitted to [AAAI2021](https://aaai.org/Conferences/AAAI-21/aaai21call/))

## Main requirements

  * **torch == 1.0.0**
  * **torchvision == 0.2.0**
  * **Python 3**

## Environmental settings
This repository is developed using python **3.6.7** on Ubuntu **16.04.5 LTS**. The CUDA nad CUDNN version is **9.0** and **7.4.1** respectively. We use **one NVIDIA 1080ti GPU card** for training and testing. Other platforms or GPU cards are not fully tested.

## Pretrain models
**The pretrain backbone (vgg, resnet) and pretrain DA DET model (ICR-CCR) will be released soon.**

## Usage
```bash
# install
cd DA_Faster_ICR_CCR/lib
python setup.py build develop
# to train FLDMN on cityscapes:
python da_trainval_net.py
# To validate FLDMN on cityscape:
python test_cityscape.py
```

## Data and Format
**The data will be released soon.**
