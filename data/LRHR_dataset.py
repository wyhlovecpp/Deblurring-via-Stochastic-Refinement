from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from medpy.io import load
import numpy as np
import scipy.io as io
class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=64, r_resolution=64, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.path = Util.get_paths_from_images(
            '{}'.format(dataroot))
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = os.path.join(self.path[index])
        image= io.loadmat(image_path)['img']
        image_h = image[:,128:256,:]
        img_hpet = torch.Tensor(image_h)
        image_s = image[:,0:128,:]
        img_spet = torch.Tensor(image_s)
        if self.need_LR:
            image_l = image[:,0:128,:]
            img_lpet = torch.Tensor(image_l)
        if self.need_LR:
            return {'LR': img_lpet, 'HR': img_hpet, 'SR': img_spet, 'Index': index}
        else:
            return {'HR': img_hpet, 'SR': img_spet, 'Index': index}