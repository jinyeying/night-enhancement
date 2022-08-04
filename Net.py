import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os

class Net(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Net, self).__init__()
        self.input_nc = input_nc

        self.conv1_1  = nn.Conv2d(input_nc, 32, 3, padding=1)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1    = nn.BatchNorm2d(32)
        self.conv1_2  = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2    = nn.BatchNorm2d(32)
        self.max_pool1= nn.MaxPool2d(2)

        self.conv2_1  = nn.Conv2d(32, 64, 3, padding=1)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1    = nn.BatchNorm2d(64)
        self.conv2_2  = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2    = nn.BatchNorm2d(64)
        self.max_pool2= nn.MaxPool2d(2)

        self.conv3_1  = nn.Conv2d(64, 128, 3, padding=1)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1    = nn.BatchNorm2d(128)
        self.conv3_2  = nn.Conv2d(128, 128, 3, padding=1)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2    = nn.BatchNorm2d(128)
        self.max_pool3= nn.MaxPool2d(2)

        self.conv4_1  = nn.Conv2d(128, 256, 3, padding=1)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1    = nn.BatchNorm2d(256)
        self.conv4_2  = nn.Conv2d(256, 256, 3, padding=1)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2    = nn.BatchNorm2d(256)
        self.max_pool4= nn.MaxPool2d(2)

        self.conv5_1  = nn.Conv2d(256, 512, 3, padding=1)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1    = nn.BatchNorm2d(512)
        self.conv5_2  = nn.Conv2d(512, 512, 3, padding=1)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2    = nn.BatchNorm2d(512)

        self.deconv5  = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_1  = nn.Conv2d(512, 256, 3, padding=1)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1    = nn.BatchNorm2d(256)
        self.conv6_2  = nn.Conv2d(256, 256, 3, padding=1)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2    = nn.BatchNorm2d(256)

        self.deconv6  = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_1  = nn.Conv2d(256, 128, 3, padding=1)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1    = nn.BatchNorm2d(128)
        self.conv7_2  = nn.Conv2d(128, 128, 3, padding=1)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2    = nn.BatchNorm2d(128)

        self.deconv7  = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_1  = nn.Conv2d(128, 64, 3, padding=1)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1    = nn.BatchNorm2d(64)
        self.conv8_2  = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2    = nn.BatchNorm2d(64)

        self.deconv8  = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_1  = nn.Conv2d(64, 32, 3, padding=1)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1    = nn.BatchNorm2d(32)
        self.conv9_2  = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10   = nn.Conv2d(32, output_nc, 1)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            xavier(m.bias.data)

    def forward(self, input):
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

        conv5 = F.interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=False)
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=False)
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=False)
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=False)
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)
        
        return latent