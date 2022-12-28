import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import sleep
import os
import argparse
import random
import skimage
import cv2
from tqdm import tqdm
from torchvision import utils as vutils
import load_data as DA
from Net import *
from guided_filter_pytorch.guided_filter import GuidedFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", type=str, default='GOPR0364_frame_000939_rgb_anon.png',
        help="Image to be used for demo")
    parser.add_argument("--out_dir", type=str,  default='./light-effects-output/',
        help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--data_dir", type=str, default='./light-effects/',
        help="Directory containing images with light-effects for demo")
    parser.add_argument("--load_model", type=str, default=None,
        help="model to initialize with")
    parser.add_argument("--load_size", type=str, default="Resize",
        help="Width and height to resize training and testing frames. None for no resizing, only [512, 512] for no resizing")
    parser.add_argument("--crop_size", type=str, default="[512, 512]",
        help="Width and height to crop training and testing frames. Must be a multiple of 16")
    parser.add_argument("--iters", type=int, default=60,
        help="No of iterations to train the model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="Learning rate for the model.")
    return parser.parse_args()

def get_LFHF(image, rad_list=[4, 8, 16, 32], eps_list=[0.001, 0.0001]):
    def decomposition(guide, inp, rad_list, eps_list):
        LF_list = []
        HF_list = []
        for radius in rad_list:
            for eps in eps_list:
                gf = GuidedFilter(radius, eps)
                LF = gf(guide, inp)
                LF[LF>1] = 1 
                LF_list.append(LF)
                HF_list.append(inp - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF
    image = torch.clamp(image, min=0.0, max=1.0)
    # Compute the LF-HF features of the image
    img_lf, img_hf = decomposition(guide=image, 
                                   inp=image, 
                                   rad_list=rad_list,
                                   eps_list=eps_list)
    return img_lf, img_hf

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
        
class Vgg16ExDark(torch.nn.Module):
    def __init__(self, load_model=None, requires_grad=False):
        super(Vgg16ExDark, self).__init__()
        # Create the model
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if load_model is None:
            print('Vgg16ExDark needs a pre-trained checkpoint!')
            raise Exception
        else:
            print('Vgg16ExDark initialized with %s'% load_model)
            model_state_dict = torch.load(load_model)
            model_dict       = self.vgg_pretrained_features.state_dict()
            model_state_dict = {k[16:]: v for k, v in model_state_dict.items() if k[16:] in model_dict}
            self.vgg_pretrained_features.load_state_dict(model_state_dict)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] 
        out = []
        for i in range(indices[-1]+1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out

class PerceptualLossVgg16ExDark(nn.Module):
    def __init__(self, vgg=None, 
                 load_model=None,
                 weights=None, 
                 indices=None, 
                 normalize=True):
        super(PerceptualLossVgg16ExDark, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16ExDark(load_model)
        else:
            self.vgg = vgg
        self.vgg     = self.vgg.cuda()
        self.criter  = nn.L1Loss()
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.indices = indices or [3, 8, 15, 22]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], 
                                       [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criter(x_vgg[i], y_vgg[i].detach())
        return loss

class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class ExclusionLoss(nn.Module):
    def __init__(self, level=3):
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

def gradient(pred):
    D_dy      = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx      = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        
    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)

def smooth_loss(pred_map):
    dx, dy   = gradient(pred_map)
    dx2, dxdy= gradient(dx)
    dydx, dy2= gradient(dy)
    loss     =  (dx2.abs().mean()  + dxdy.abs().mean()+ 
                 dydx.abs().mean() + dy2.abs().mean())
    return loss

def rgb2gray(rgb):
    gray = 0.2989*rgb[:,:,0:1,:] + \
    	   0.5870*rgb[:,:,1:2,:] + \
    	   0.1140*rgb[:,:,2:3,:]
    return gray

def validate(dle_net, 
             inputs):
    print('Validation not possible since there are no labels!')
    raise Exception

def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)

def calc_psnr_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y[mask], im2_y[mask])

def calc_ssim_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y[mask], im2_y[mask])

def demo(args, 
		  dle_net, 
          optimizer_dle_net, 
          inputs):
    
    dle_net.train()

    img_in    = Variable(torch.FloatTensor(inputs['img_in'])).cuda()
    optimizer_dle_net.zero_grad()

    le_pred = dle_net(img_in)
    dle_pred= img_in + le_pred

    lambda_cc         = 1.0 
    dle_pred_cc       = torch.mean(dle_pred, dim=1, keepdims=True)
    cc_loss           = (F.l1_loss(dle_pred[:, 0:1, :, :], dle_pred_cc) + \
                         F.l1_loss(dle_pred[:, 1:2, :, :], dle_pred_cc) + \
                         F.l1_loss(dle_pred[:, 2:3, :, :], dle_pred_cc))*(1/3) ##Color Constancy Loss

    lambda_recon        = 1.0
    recon_loss          = F.l1_loss(dle_pred, img_in)                         
    
    lambda_excl        = 0.01
    data_type          = torch.cuda.FloatTensor
    excl_loss          = ExclusionLoss().type(data_type)                      

    lambda_smooth       = 1.0 
    le_smooth_loss      = smooth_loss(le_pred)

    loss = lambda_recon*recon_loss + \
           lambda_cc*cc_loss
    loss += lambda_excl * excl_loss(dle_pred, le_pred)
    loss += lambda_smooth*le_smooth_loss
    loss.backward()

    optimizer_dle_net.step()

    imgs_dict   = {}
    imgs_dict['dle_pred'] = dle_pred.detach().cpu()
    return imgs_dict

if __name__ == '__main__':
    args = get_arguments()

    args.imgin_dir = args.data_dir
    args.use_gray  = False

    torch.manual_seed(0)

    args.imgs_dir = args.out_dir
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)

    if args.use_gray:
        channels = 1
    else:
        channels = 3
    dle_net = Net(input_nc=channels, output_nc=channels)
    dle_net = nn.DataParallel(dle_net).cuda()

    if args.load_model is not None:
        dle_net_ckpt_file = args.load_model
        dle_net.load_state_dict(torch.load(dle_net_ckpt_file)['state_dict'])

    optimizer_dle_net = optim.Adam(dle_net.parameters(), 
                                      lr=args.learning_rate, 
                                      betas=(0.9, 0.999))

    da_list  = sorted([(args.imgin_dir+ file) for file in os.listdir(args.imgin_dir) \
                            if file == args.img_name])
    demo_list   = da_list
    demo_list   = demo_list*args.iters

    Dele_Loader  = torch.utils.data.DataLoader(DA.loadImgs(args, 
                                                           demo_list,
                                                           mode='demo'),
                                               batch_size  = 1, 
                                               shuffle     = True, 
                                               num_workers = 16, 
                                               drop_last   = False)
    count_idx = 0
    tbar = tqdm(Dele_Loader)
    for batch_idx, inputs in enumerate(tbar):
        count_idx = count_idx + 1
        imgs_dict = demo(args,
                          dle_net, 
                          optimizer_dle_net,
                          inputs)
        tbar.update()

        if (count_idx%60 == 0):
            inout = os.path.join(args.imgs_dir, args.img_name[:-4]+'_in_out')
            out   = os.path.join(args.imgs_dir, args.img_name[:-4]+'_out')
            save_img   = torch.cat((inputs['img_in'][0, :, :, :],
                                    imgs_dict['dle_pred'][0, :, :, :]), dim=2)
            out_img    = imgs_dict['dle_pred'][0, :, :, :]
            vutils.save_image(save_img, inout+'.png')
            vutils.save_image(out_img, out+'.png')
