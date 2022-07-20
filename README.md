# night_enhancement (ECCV'2022)
Implementation of paper "Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression" (ECCV'2022)

## Introduction
This is an implementation of the following paper.
> Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression
> European Conference on Computer Vision (ECCV'2022)

Yeying Jin, [Wenhan Yang](https://flyywh.github.io/) and [Robby T. Tan](https://tanrobby.github.io/pub.html)


## Datasets
### Light-Effects Suppression on Night Data
1. [Light-effects data](https://www.dropbox.com/sh/ro8fs629ldebzc2/AAD_W78jDffsJhH-smJr0cNSa?dl=0) <br>
Light-effects data is collected from Flickr and by ourselves, with multiple light colors in various scenes: Aashish Sharma, Robby T. Tan. "Nighttime Visibility Enhancement by Increasing the Dynamic Range and Suppression of Light Effects", CVPR, 2021.


2. [LED data](https://www.dropbox.com/sh/7lhpnj2onb8c3dl/AAC-UF1fvJLxvCG-IuYLQ8T4a?dl=0) <br>
We captured images with dimmer light as the reference images.


3. [GTA5](https://www.dropbox.com/sh/gfw44ttcu5czrbg/AACr2GZWvAdwYPV0wgs7s00xa?dl=0) <br>
Synthetic GTA5 nighttime fog data: Wending Yan, Robby T. Tan, Dengxin Dai. "Nighttime Defogging Using High-Low Frequency Decomposition and Grayscale-Color Networks", ECCV, 2020.

4. [Syn-light-effects](https://www.dropbox.com/sh/2sb9na4ur7ry2gf/AAB1-DNxy4Hq6qPU-afYIKVaa?dl=0) <br>
Synthetic-light-effects data is the implementation of the paper, S. Metari, F. Deschênes, "A New Convolution Kernel for Atmospheric Point Spread Function Applied to Computer Vision", ICCV, 2017.

### Low-Light Enhancement
1. [LOL dataset](https://daooshee.github.io/BMVC2018website/) <br>
LOL: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement", BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing) <br>

2. [LOL-Real dataset](https://github.com/flyywh/CVPR-2020-Semi-Low-Light/) <br>
LOL-real (the extension work): Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [[Google Drive]](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) <br> <br>
We use LOL-real as it is larger and more diverse.


## Low-Light Enhancement Results:
### Pre-trained Model

1. Download the [pre-trained LOL model](https://www.dropbox.com/s/0ykpsm1d48f74ao/LOL_params_0900000.pt?dl=0), put in results/LOL/model/

### Results
1. [LOL-Real Results](https://www.dropbox.com/sh/t6eb4aq025ctnhy/AADRRJNN3u-N8HApe1tFo19Ra?dl=0)<br>

Get the following Table 4 in the main paper on the LOL-Real dataset (100 test images).
|Learning| Method | PSNR | SSIM | 
|--------|--------|------|------ |
| Unsupervised Learning| **Ours** | **25.51** |**0.8015**|
| N/A | Input | 9.72 | 0.1752|


2. [LOL-test Results](https://www.dropbox.com/sh/la21ocjk14dtg9t/AABOBsCQ39Oml33fItqX5koFa?dl=0)<br>

Get the following Table 3 in the main paper on the LOL-test dataset (15 test images).
|Learning| Method | PSNR | SSIM | 
|--------|--------|------|------ |
| Unsupervised Learning| **Ours** | **21.521** |**0.7647**|
| N/A | Input | 7.773 | 0.1259|

## VGG Results:

1. Download the [fine-tuned VGG model](https://www.dropbox.com/s/xzzoruz1i6m7mm0/model_best.tar?dl=0) (fine-tuned on ExDark dataset), put in 
VGG_code/ckpts/vgg16_featureextractFalse_ExDark/nets/


### Citation
If light-effects data is useful for your research, please cite our paper. 
```
@inproceedings{sharma2021nighttime,
	title={Nighttime Visibility Enhancement by Increasing the Dynamic Range and Suppression of Light Effects},
	author={Sharma, Aashish and Tan, Robby T},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	pages={11977--11986},
	year={2021}
}
```

If GTA5 nighttime fog data is useful for your research, please cite our paper. 
```
@inproceedings{yan2020nighttime,
	title={Nighttime defogging using high-low frequency decomposition and grayscale-color networks},
	author={Yan, Wending and Tan, Robby T and Dai, Dengxin},
	booktitle={European Conference on Computer Vision},
	pages={473--488},
	year={2020},
	organization={Springer}
}
```
