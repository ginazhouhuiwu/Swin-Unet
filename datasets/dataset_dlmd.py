import glob, os

import numpy as np

import torch
from torch.utils.data import Dataset

from scipy import ndimage
from scipy.ndimage.interpolation import zoom


file_path_diffuser = '../../2tb_ssd/dlmd/diffuser_images/'
file_path_lensed = '../../2tb_ssd/dlmd/ground_truth_lensed/'

diffuser_files = glob.glob(file_path_diffuser + '/*.npy')
lensed_files = glob.glob(file_path_lensed + '/*.npy')
assert len(diffuser_files) == 24999, len(lensed_files) == 24999

diffuser_files.sort()
lensed_files.sort()

def preprocess(image):
    # output_shape = (3, 210, 380)
    image = image.transpose(1, 2, 0)
    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1]  = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]
    out_image = np.flipud(np.clip(image_color, 0, 1))
    return out_image[60:, 62:-38, :].transpose(2, 0, 1)

def crop_center_square(img):
    # output_shape = (3, 192, 192)
    starty = img.shape[1]//2 - 192//2 
    startx = img.shape[2]//2 - 192//2 
    return img[:, starty:starty + 192, startx:startx + 192]

def center_pad(img):
    # output_shape = (3, 224, 224)
    pad = (224 - img.shape[1]) // 2
    return np.pad(img, ((0, 0), (pad, pad), (pad, pad)), constant_values=(0, 0))

def crop_pad(img):
    # output_shape = (3, 192, 192)
    pad = (img.shape[1] - 192) // 2
    return img[..., pad:-pad, pad:-pad]