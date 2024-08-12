"""
Developed on transparency_separation.py 
"""
from net import skip
from net.losses import *
from net.noise import get_noise
from utils.image_io import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import os
import tqdm
from collections import namedtuple

class LeSeparation(object):
    def __init__(self, 
                 image_name, 
                 image,
                 output_path,
                 plot_during_training=True, 
                 show_every=200, 
                 num_iter=8000,
                 original_layer1=None, 
                 original_layer2=None):
        self.image                = image
        self.plot_during_training = plot_during_training
        self.use_cc_loss         = True # Newly added
        self.use_le_smooth_loss  = True # Newly added

        self.psnrs      = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter   = num_iter
        self.loss_function = None
        self.output_path   = output_path
        self.parameters    = None
        self.learning_rate = 0.1# 0.001 default
        self.input_depth   = 3
        self.layer1_net_inputs = None
        self.layer2_net_inputs = None
        self.layer1_isle   = None
        self.original_layer1 = original_layer1
        self.original_layer2 = original_layer2
        self.layer1_net = None
        self.layer2_net = None
        self.total_loss = None
        self.layer1_out = None
        self.layer2_out = None
        self.current_result = None
        self.best_result    = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_datarefs()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images          = create_augmentations(self.image)
        self.images_torch    = [np_to_torch(image).type(torch.cuda.FloatTensor) \
                                     for image in self.images]

    def _init_datarefs(self):
    	pass

    def _init_inputs(self):
        input_type   = 'noise'
        # input_type = 'meshgrid'
        data_type    = torch.cuda.FloatTensor
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                                  input_type,
                                                  (self.images_torch[0].shape[2],
                                                   self.images_torch[0].shape[3])).type(data_type).detach())
        self.layer1_net_inputs = [np_to_torch(aug).type(data_type).detach() \
                                  for aug in create_augmentations(origin_noise)]
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.layer2_net_inputs = [np_to_torch(aug).type(data_type).detach() \
                                  for aug in create_augmentations(origin_noise)]

    def _init_parameters(self):
        self.parameters = [p for p in self.layer1_net.parameters()] + \
                          [p for p in self.layer2_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'layer1'
        layer1_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.layer1_net = layer1_net.type(data_type)

        layer2_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.layer2_net = layer2_net.type(data_type)

        # layer3_net = skip(
        #     input_depth, 1,
        #     num_channels_down=[8, 16, 32, 64, 128],
        #     num_channels_up=[8, 16, 32, 64, 128],
        #     num_channels_skip=[0, 0, 0, 4, 4],
        #     upsample_mode='bilinear',
        #     need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        # self.layer3_net = layer3_net.type(data_type)

    def _init_losses(self):
        data_type      = torch.cuda.FloatTensor
        self.l1_loss   = nn.L1Loss().type(data_type)
        self.excl_loss = ExclusionLoss().type(data_type)
        self.le_smooth_loss = smooth_loss
        self.cc_loss   = cc_loss

    def optimize(self):
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = True

        optimizer  = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        time_start = time.time()
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j, time_start)
            optimizer.step()

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(step)
        if step == self.num_iter - 1:
            aug = 0
        self.aug= aug
        layer1_net_input = self.layer1_net_inputs[aug]  + \
                           (self.layer1_net_inputs[aug].clone().normal_() * reg_noise_std)
        layer2_net_input = self.layer2_net_inputs[aug] + \
                           (self.layer2_net_inputs[aug].clone().normal_() * reg_noise_std)


        ###########################################################################################
        """
        Noisy input images can also be inputted, 
        But, this needs adjustment of the weights of the losses used below 
        """
        self.layer1_out = self.layer1_net(layer1_net_input)# + self.images_torch[aug])
        self.layer2_out = self.layer2_net(layer2_net_input)# + self.images_torch[aug])

        self.total_loss = self.l1_loss(self.layer1_out + self.layer2_out, self.images_torch[aug])  ##Reconstruction Loss
        self.total_loss += 0.01 * self.excl_loss(self.layer1_out, self.layer2_out)                 ##Gradient Exlusion Loss 
        ###########################################################################################


        ###########################################################################################
        sigma        = 0.35
        image_minrgb = torch.min(self.images_torch[aug], dim=1, keepdim=True)[0]
        le_mask = torch.exp(-(1.0 - image_minrgb)**2/(2*sigma**2))                                 ##Gaussian Mask
        le_mask = torch.cat((le_mask, le_mask, le_mask), dim=1)
        le_mask = le_mask>0.3

        layer1_distance = torch.mean((self.layer1_out[le_mask].clone() - 
                                      self.images_torch[aug][le_mask]).abs()).detach().item()
        layer2_distance = torch.mean((self.layer2_out[le_mask].clone() - 
                                      self.images_torch[aug][le_mask]).abs()).detach().item()
        if layer1_distance<layer2_distance:
            self.layer1_isle = True
            self.le_mask  = le_mask
        else:
            self.layer1_isle = False 
            self.le_mask  = le_mask
        ###########################################################################################


        ###########################################################################################
        if self.use_cc_loss:                                                                      ##Color Constancy Loss
            """
            800<step<=4000: 0.07/0.1, step>4000: 0.01
            """
            if step>800 and step<=4000:
                if self.layer1_isle:
                    self.total_loss += 0.07 * self.cc_loss(self.layer2_out) # Use 0.1 weight?
                else:
                    self.total_loss += 0.07 * self.cc_loss(self.layer1_out) # Use 0.1 weight?
            if step>4000:
                if self.layer1_isle:
                    self.total_loss += 0.01 * self.cc_loss(self.layer2_out)
                else:
                    self.total_loss += 0.01 * self.cc_loss(self.layer1_out)
        ###########################################################################################


        ###########################################################################################
        if self.use_le_smooth_loss:
            """
            step>2000: 1.0
            """
            if step>2000:
                if self.layer1_isle:
                    self.total_loss += 1.0 * self.le_smooth_loss(self.layer1_out)
                else:
                    self.total_loss += 1.0 * self.le_smooth_loss(self.layer2_out)
        ###########################################################################################

        # Backprop the total loss
        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            self.input_np          = np.clip(self.images[self.aug], 0, 1)
            self.layer1_out_np     = np.clip(torch_to_np(self.layer1_out), 0, 1)
            self.layer2_out_np     = np.clip(torch_to_np(self.layer2_out), 0, 1)
            self.le_mask_np        = np.clip(torch_to_np(self.le_mask), 0, 1)
            self.reconstructed_np  = np.clip(self.layer1_out_np+self.layer2_out_np, 0, 1)
            self.psnr = compare_psnr(self.images[self.aug], self.reconstructed_np)
            self.psnrs.append(self.psnr)
            if self.layer1_isle:
                self.le_np = self.layer1_out_np
                self.back_np = self.layer2_out_np
            else:
                self.le_np = self.layer2_out_np
                self.back_np = self.layer1_out_np
            self.back_o_np  = np.clip((self.input_np - self.le_np), 0, 1)

    def _plot_closure(self, step, time_start):
        print('Iteration:{:5d}  Time:{:2f}mins  Loss:{:5f}  PSNR (Recon):{:2f} IsLayer1Le:{}'.format(step,
              (time.time()-time_start)/60,
              self.total_loss.item(),
              self.psnr,
              self.layer1_isle),
              '\r', end='')
        if step % self.show_every == self.show_every - 1:
            output_image = np.concatenate((self.input_np, 
                                           self.back_o_np), axis=2)

            save_image(self.image_name + "_in_out_{}".format(step), 
                       output_image,
                       self.output_path)
            
    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs, self.output_path)
        #save_image(self.image_name + "_original", self.images[self.aug], self.output_path)

if __name__ == "__main__":
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    
    parser = argparse.ArgumentParser(description="Decompose the input image into background and light-effects layers.")
    parser.add_argument('--img_name', type=str, default='DSC01065.JPG', help="Image to be used for demo")
    parser.add_argument('--out_dir', type=str, default='./light-effects-output/', help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--data_dir", type=str, default='./light-effects/',help="Directory containing images with light-effects for demo")
    
    args = parser.parse_args()

    args.imgin_dir = args.data_dir
    args.imgs_dir = args.out_dir
    args.output_path = os.path.join(args.imgs_dir, os.path.splitext(args.img_name)[0])
    os.makedirs(args.output_path, exist_ok=True)

    I = prepare_image(args.imgin_dir+args.img_name)
    s = LeSeparation(os.path.splitext(args.img_name)[0], I , args.output_path)
    s.optimize()
    s.finalize()
