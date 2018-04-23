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
        data_list.sort(key= lambda x:int(x[:-6]))
        #print (data_list)
        num_sample=len(data_list)
        if test:
            self.data_list=data_list[int(0.9*num_sample):]
            print (self.data_list[0:9])
            print ('test dataset ,len={}\n'.format(len(self.data_list)))
        else:
            self.data_list=data_list[:int(0.9*num_sample)]
            print (self.data_list[-8:])
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
        pos_coord=dense_to_sparse(voxel)
        pos_coord=64-pos_coord
        voxel=sparse_to_dense(pos_coord,64)
        img = self.transf(img_data[:,:,:3])

        return img, voxel

    def pull_img(self,index):
        data = cv2.imread(os.path.join(self.data_rootpath, self.data_list[index]), -1)
        if data is None:
            print (self.data_list[index])

        img_data = data[:, :256, :]
        img = self.transf(img_data[:, :, :3])

        return self.data_list[index],img

    def pull_target(self,filename):
        data = cv2.imread(os.path.join(self.data_rootpath, filename), -1)
        voxel_data = data[:, 256:, :]
        voxel = (self.unpack_voxels(voxel_data, 64, 4)).astype(np.bool)

        return voxel





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




"""
dataset=singleDataset('../dataset/chairs/database_64/')
print (dataset.data_list)
for i in xrange(8):
    dataset2 = singleDataset('../dataset/chairs/database_64/', test=True)
    print(dataset2.data_list)
"""
def generate_datalist(model_list):
    data_list=[]
    for model in model_list:
        for i in xrange(8):
            for j in xrange(8):
                if i==j:
                    continue
                data_list.append('{}_{}_{}'.format(model,i,j))
    return data_list


## dataset class for MULTI-view CNN training
class multiDataset(data.Dataset):
    def __init__(self, data_root, num_model, transf=torchvision.transforms.ToTensor(), test=False):
        super(multiDataset, self).__init__()
        self.transf = transf
        self.data_rootpath=data_root

        model_list=np.arange(num_model).tolist()
        if test:
            data_list= model_list[int(399 * num_model/400):]
            self.data_list=generate_datalist(data_list)
            print (self.data_list[0:9])
            print ('test dataset ,len={}\n'.format(len(self.data_list)))
        else:
            data_list= model_list[:int(399 * num_model/400)]
            self.data_list = generate_datalist(data_list)
            print (self.data_list[-8:])
            print('training dataset ,len={}\n'.format(len(self.data_list)))


    def __getitem__(self, index):
        img1, img2, gt1, gt2, v12 = self.pull_item(index)

        return img1, img2, gt1, gt2, v12

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

    def unpack_data(self,data,id):
        if data is None:
            print (id)

        img_data = data[:, :256, :]
        voxel_data = data[:, 256:, :]
        voxel = (self.unpack_voxels(voxel_data, 64, 4)).astype(np.bool)
        img = self.transf(img_data[:,:,:3])

        return img,voxel


    def pull_item(self,index):
        model_idx, v1_idx, v2_idx = self.data_list[index].split('_')

        img1_id = '{}_{}'.format(model_idx, v1_idx)
        img2_id = '{}_{}'.format(model_idx, v2_idx)

        data1 = cv2.imread(os.path.join(self.data_rootpath,img1_id + '.png'), -1)
        data2 = cv2.imread(os.path.join(self.data_rootpath,img2_id + '.png'), -1)

        img1,voxel1=self.unpack_data(data1,img1_id)
        img2,voxel2=self.unpack_data(data2,img2_id)

        return img1,img2, voxel1,voxel2,(v1_idx,v2_idx)

    def pull_img(self,index):
        data = cv2.imread(os.path.join(self.data_rootpath, self.data_list[index]), -1)
        if data is None:
            print (self.data_list[index])

        img_data = data[:, :256, :]
        img = self.transf(img_data[:, :, :3])

        return self.data_list[index],img

    def pull_target(self,filename):
        data = cv2.imread(os.path.join(self.data_rootpath, filename), -1)
        voxel_data = data[:, 256:, :]
        voxel = (self.unpack_voxels(voxel_data, 64, 4)).astype(np.bool)

        return voxel


def multi_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    binvox model.

    Arguments:
        batch: the batch of :(tuple) A tuple of tensor images and annotations(.binvox data)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    target1s = []
    target2s = []
    img1s = []
    img2s = []
    v12s = []
    for sample in batch:
        img1s.append((sample[0]))
        img2s.append((sample[1]))

        target1s.append(torch.from_numpy((sample[2].astype(np.float32))).float())
        target2s.append(torch.from_numpy((sample[3].astype(np.float32))).float())

        v12s.append((sample[4]))
    return torch.stack(img1s, 0), torch.stack(img2s, 0), \
           torch.stack(target1s, 0), torch.stack(target2s, 0), v12s


