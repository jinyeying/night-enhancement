import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy.interpolate import interp1d

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def rgb_loader(img_path):
    assert(is_image_file(img_path)==True)
    return Image.open(img_path).convert('RGB')

def gray_loader(img_path):
    assert(is_image_file(img_path)==True)
    return Image.open(img_path).convert('L')

class loadImgs(data.Dataset):
    def __init__(self, 
                 args, 
                 imgin_list, 
                 mode='demo'):
        self.imgin_list  = imgin_list
        self.args        = args
        self.mode        = mode
        
        if self.args.use_gray:
            self.img_loader  = gray_loader
        else:
            self.img_loader  = rgb_loader

        self.data_list   = {'img_in': []}
        random.seed(141)
        for num_img in range(len(self.imgin_list)):
            self.data_list['img_in'].append(self.imgin_list[num_img])

    def __getitem__(self, index):
        img_in_path = self.data_list['img_in'][index]

        if self.mode == 'demo':
            img_in  = self.img_loader(img_in_path)
            if self.args.load_size != 'None':
                w, h      = img_in.size
                img_in    = img_in.resize((512, 512))
            if self.args.crop_size != 'None':
                w, h      = img_in.size
                crop_size = self.args.crop_size.strip('[]').split(', ')
                crop_size = [int(item) for item in crop_size]
                th, tw    = crop_size[0], crop_size[1]
                x1        = random.randint(0, w - tw)
                y1        = random.randint(0, h - th)
                img_in    = img_in.crop((x1, y1, x1 + tw, y1 + th))
        elif self.mode == 'val':
            raise NotImplementedError
        elif self.mode == 'predict':
            img_in  = self.img_loader(img_in_path)
        else:
            print('Unrecognized mode! Please select among: (demo, val, predict)')
            raise NotImplementedError

        t_list = [transforms.ToTensor()]
        composed_transform  = transforms.Compose(t_list)
        if self.mode == 'demo':
            img_in = composed_transform(img_in)
        if self.mode == 'val':
            raise NotImplementedError
        if self.mode == 'predict':
            img_in = composed_transform(img_in)

        if self.mode == 'demo':
            inputs = {'img_in': img_in}
            return inputs
        if self.mode == 'predict':
            inputs = {'img_in': img_in}
    def __len__(self):
        return len(self.data_list['img_in'])
