import glob, os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from scipy import ndimage
from scipy.ndimage.interpolation import zoom


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
    pad = (img.shape[1] - 192) // 2
    return img[..., pad:-pad, pad:-pad]

class DLMDDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        Args:
            file_list (string): List of .npy file paths.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.file_list[idx]
        image_np = np.load(img_path).transpose((2, 0, 1))  # Load image as numpy array
        image_np = center_pad(crop_center_square(preprocess(image_np)))
            
        # Convert image from numpy array to tensor
        image = torch.from_numpy(image_np.copy())

        if self.transform:
            image = self.transform(image)

        return image
    
def split():
    file_path_diffuser = '/home/gwu_waller/2tb_ssd/dlmd/diffuser_images'
    file_path_lensed = '/home/gwu_waller/2tb_ssd/dlmd/ground_truth_lensed'

    diffuser_files = glob.glob(file_path_diffuser + '/*.npy')
    lensed_files = glob.glob(file_path_lensed + '/*.npy')
    assert len(diffuser_files) == 24999, len(lensed_files) == 24999

    diffuser_files.sort()
    lensed_files.sort()

    N_train = 24900
    trainfiles_images, valfiles_images= diffuser_files[:N_train], diffuser_files[N_train:]
    trainfiles_labels, valfiles_labels = lensed_files[:N_train], lensed_files[N_train:]

    traindata_images = DLMDDataset(file_list=trainfiles_images)
    traindata_labels = DLMDDataset(file_list=trainfiles_labels)

    trainloader_images = DataLoader(traindata_images, batch_size=1, shuffle=False, num_workers=4)
    trainloader_labels = DataLoader(traindata_labels, batch_size=1, shuffle=False, num_workers=4)

    valdata_images = DLMDDataset(file_list=valfiles_images)
    valdata_labels = DLMDDataset(file_list=valfiles_labels)

    valloader_images = DataLoader(valdata_images, batch_size=1, shuffle=False, num_workers=4)
    valloader_labels = DataLoader(valdata_labels, batch_size=1, shuffle=False, num_workers=4)

    return trainloader_images, trainloader_labels, valloader_images, valloader_labels
