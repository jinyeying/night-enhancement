from __future__ import print_function
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import inspect, re
import os
import collections
from torchvision import transforms

# def normalize():
#     return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# def inv_normalize():
#     return transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
def normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
def normalize_grayscale():
    return transforms.Normalize([0.5], [0.5])
def inv_normalize():
    return transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5])
def inv_normalize_grayscale():
    return transforms.Normalize([-0.5/0.5], [1/0.5])
			 
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def imgtensor2im(image_tensor, imtype=np.uint8):
    image_numpy = inv_normalize(image_tensor).cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    if image_numpy.shape[2] < 3:
        image_numpy = np.dstack([image_numpy]*3)
    return image_numpy.astype(imtype)

def disptensor2im(image_tensor, maxdisp, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0/maxdisp
    if image_numpy.shape[2] < 3:
        image_numpy = np.dstack([image_numpy]*3)
    return image_numpy.astype(imtype)

def gkern_2d(size=5, sigma=3):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size//2, size//2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * 3)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
