import os
import torch
from torch import nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import numpy as np
from torchvision import transforms as transforms
from torchvision import utils as vutils
from PIL import Image
from models.networks import Vgg16ExDark, MeanShift
import matplotlib.pyplot as plt

def main():
    # Get the VGG16 network
    vggclass_dir            = './'
    exdark_ft_vgg16net_ckpt = vggclass_dir + '/ckpts/vgg16_featureextractFalse_ExDark/nets/model_best.tar'
    nets_vgg      = Vgg16ExDark(exdark_ft_vgg16net_ckpt)
    nets_vgg      = nn.DataParallel(nets_vgg, device_ids=[0]).cuda()
    nets_vgg_norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
    
    # Get data
    in1_nightL     = process_img('./results_VGGfeatures/DSC01607_input.jpg')
    in1_nightL     = (in1_nightL+1)*0.5
    in1_nightL_gray= process_img_gray('./results_VGGfeatures/DSC01607_GrayBest.png')
    in1_nightL_gray= (in1_nightL_gray+1)*0.5
    in1_nightL_gray= torch.cat((in1_nightL_gray, in1_nightL_gray, in1_nightL_gray), dim=1)
    
    out1_nightL    = process_img('./results_VGGfeatures/DSC01607_Jrefine.jpg')
    out1_nightL    = (out1_nightL+1)*0.5

    # Save checks dir
    savechecks_dir = './results_VGGfeatures/DSC01607/'
    os.makedirs(savechecks_dir, exist_ok=True)

    # Prepare data for VGG
    in1_nightL_vgg  = nets_vgg_norm(in1_nightL)
    out1_nightL_vgg = nets_vgg_norm(out1_nightL)

    # Check features from VGG's num_layer
    num_layer       = [15] # Fix to 15! 
    in1_nightL_fts  = nets_vgg(in1_nightL_vgg, num_layer)
    out1_nightL_fts = nets_vgg(out1_nightL_vgg,  num_layer)
    assert(len(in1_nightL_fts)==1)
    assert(len(out1_nightL_fts) ==1)
    in1_nightL_fts  = in1_nightL_fts[0]
    out1_nightL_fts = out1_nightL_fts[0]
    num_fts         = in1_nightL_fts.size()[1]
    for num_ft in range(num_fts):
        size_input    = (in1_nightL.size()[2], in1_nightL.size()[3])
        in1_nightL_ft = F.interpolate((in1_nightL_fts[:, num_ft:num_ft+1, :, :]), 
                                     size_input)
        out1_nightL_ft= F.interpolate((out1_nightL_fts[:,  num_ft:num_ft+1, :, :]), 
                                     size_input)
        in1_nightL_ft = torch.div(in1_nightL_ft - torch.min(in1_nightL_ft), 
                                 torch.max(in1_nightL_ft) - torch.min(in1_nightL_ft))
        out1_nightL_ft= torch.div(out1_nightL_ft - torch.min(out1_nightL_ft), 
                                 torch.max(out1_nightL_ft) - torch.min(out1_nightL_ft))
        cmap          = plt.get_cmap('jet')
        in1_nightL    = in1_nightL.detach().cpu()
        in1_nightL_gray= in1_nightL_gray.detach().cpu()
        out1_nightL   = out1_nightL.detach().cpu()
        in1_nightL_ft = torch.FloatTensor(cmap(in1_nightL_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]).permute(2,0,1).unsqueeze(0)
        out1_nightL_ft= torch.FloatTensor(cmap(out1_nightL_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]).permute(2,0,1).unsqueeze(0)
        # saveimg     = torch.cat((in1_nightL_gray, in1_nightL_ft, out1_nightL, out1_nightL_ft), dim=3)
        savename      = savechecks_dir + '/ft_' + '%03d'%num_ft + '_in.jpg'
        vutils.save_image(in1_nightL_ft, savename)
        savename      = savechecks_dir + '/ft_' + '%03d'%num_ft + '_out.jpg'
        vutils.save_image(out1_nightL_ft, savename)


# Some functions here
def gray2rgb(inimg):
    outimg = torch.cat((inimg, inimg, inimg), 1)
    return outimg

def normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

def normalize_gray():
    return transforms.Normalize(mean=[0.5],
                                std=[0.5])

def inv_normalize():
    return transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                                std=[1 / 0.5, 1 / 0.5, 1 / 0.5])

def inv_normalize_gray():
    return transforms.Normalize(mean=[-0.5 / 0.5],
                                std=[1 / 0.5])

def process_img(fname):
    # Setup functions
    norm_  = normalize()
    totens_= transforms.ToTensor()
    # Load and normalize images
    imgL_o = Image.open(fname).convert('RGB')
    imgL   = norm_(totens_(imgL_o)).numpy()
    imgL   = torch.FloatTensor(np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])).cuda()
    return imgL

def process_img_gray(fname):
    # Setup functions
    norm_  = normalize_gray()
    totens_= transforms.ToTensor()
    # Load and normalize images
    imgL_o = Image.open(fname).convert('L')
    imgL   = norm_(totens_(imgL_o)).numpy()
    imgL   = torch.FloatTensor(np.reshape(imgL, [1, 1, imgL.shape[1], imgL.shape[2]])).cuda()
    return imgL

if __name__ == '__main__':
    main()
