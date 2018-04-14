from __future__ import print_function,division

import os
import os.path
import torch
import torch.utils.data as data
import torchvision

import cv2

from utils.utils_trans import *
from utils.utils_rw import *

## dataset class for single-view CNN training
class singleDataset(data.Dataset):
    def __init__(self,data_root,transf=torchvision.transforms.ToTensor(),test=False):
        super(singleDataset, self).__init__()
        self.transf = transf
        self.data_rootpath=data_root

        data_list=os.listdir(data_root)
        num_sample=len(data_list)
        if test:
            self.data_list=data_list[int(0.9*num_sample):]
            print ('test dataset ,len={}\n'.format(len(self.data_list)))
        else:
            self.data_list=data_list[:int(0.9*num_sample)]
            print('training dataset ,len={}\n'.format(len(self.data_list)))


    def __getitem__(self, index):
        im, gt = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.data_list)

    def unpack_voxels(self,voxel_data, vox_size, nrows):
        ncols = vox_size / nrows / 4

        vox = voxel_data[0:vox_size, 0:vox_size, :]
        for i in range(0, int(nrows) * vox_size, vox_size):
            for j in range(0, int(ncols) * vox_size, vox_size):
                vox = np.concatenate([vox, voxel_data[i:i + vox_size, j:j + vox_size, :]], 2)
        vox = vox[:, :, 4:]

        return vox.transpose(2, 0, 1)


    def pull_item(self,index):
        data = cv2.imread(os.path.join(self.data_rootpath,self.data_list[index]), -1)
        if data is None:
            print (self.data_list[index])

        img_data = data[:, :256, :]
        voxel_data = data[:, 256:, :]

        voxel = (self.unpack_voxels(voxel_data, 64, 4)).astype(np.bool)
        img = self.transf(img_data[:,:,:3])

        return img, voxel

    def pull_img(self,index):
        data = cv2.imread(os.path.join(self.data_rootpath, self.data_list[index]), -1)
        if data is None:
            print (self.data_list[index])

        img_data = data[:, :256, :]
        img = self.transf(img_data[:, :, :3])

        return self.data_list[index],img



def single_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    binvox model.

    Arguments:
        batch: the batch of :(tuple) A tuple of tensor images and annotations(.binvox data)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append((sample[0]))
        targets.append(torch.from_numpy((sample[1].astype(np.float32))).float())

    return torch.stack(imgs, 0), torch.stack(targets,0)











