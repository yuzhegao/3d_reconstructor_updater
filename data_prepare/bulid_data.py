import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import cv2
import numpy as np
import random

from utils.utils_trans import *
from utils.utils_rw import *


'''
class AnnotationTransform(object) :
    def __init__(self):
        pass
    def __call__(self, ):
'''
## dataset class for single-view CNN training
class singleDataset(data.Dataset):
    def __init__(self,data_root,test=False):
        super(singleDataset, self).__init__()
        if test:
            data_root=os.path.join(data_root,'test')
        else:
            data_root=os.path.join(data_root,'trainval')

        self.img_path=os.path.join(data_root,'img')
        self.target_path=os.path.join(data_root,'binvox')

        self.idx=list()
        txtfile = 'trainval_csg_s.txt'
        if test :
            txtfile='test_csg_s.txt'

        for line in open(os.path.join(data_root,txtfile)):
            self.idx.append(line.strip())

    def __getitem__(self, index):
        im, gt = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.idx)


    def pull_item(self,index):
        """
        :param
        index: the index of data (img and model)

        :return: tuple of (img,binvox) img:[1x256x256]  binvox:[64x64x64]
        """
        img_id = self.idx[index] ##e.g img_id '0_1'

        with open(os.path.join(self.target_path , img_id+'.binvox')) as f:
            m = read_as_3d_array(f)
        target=m.data.transpose(2,1,0)
        img = cv2.imread(os.path.join(self.img_path , img_id+'.jpg'),0)
        if img is None:
            print img_id

        return torch.from_numpy(img).type(torch.FloatTensor),target

    def pull_img(self,index):
        img_id = self.idx[index]
        img = cv2.imread(os.path.join(self.img_path, img_id + '.jpg'), 0)
        #
        return img_id,img

    def pull_target(self,index):
        img_id = self.idx[index]
        with open(os.path.join(self.target_path , img_id+'.binvox')) as f:
            m = read_as_3d_array(f)
        target=m.data.transpose(2,1,0)
        return target

    def pull_tensor(self,index):
        torch.Tensor(self.pull_img(index)).unsqueeze_(0)


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
        imgs.append((sample[0])[np.newaxis, :])
        targets.append(torch.FloatTensor( (sample[1].astype(np.float32)) ))
    return torch.stack(imgs, 0), targets

## dataset for multi-view CNN training
class multiDataset(data.Dataset):
    def __init__(self, data_root,test=False):
        super(multiDataset, self).__init__()
        self.img_path = os.path.join(data_root, 'img')
        self.target_path = os.path.join(data_root, 'binvox')

        self.idx=list()
        txtfile = 'trainval_cube_m.txt'
        if test:
            txtfile = 'test2_cube_m.txt'

        for line in open(os.path.join(data_root, txtfile)):
            self.idx.append(line.strip())
        print len(self.idx)


    def __getitem__(self, index):
        img1, img2, gt1,gt2,v12 = self.pull_item(index)

        return img1, img2, gt1,gt2,v12

    def __len__(self):
        return len(self.idx)

    def pull_item(self, index):
        """
        input the index of model and output img x2 and .binvox x1

        index: the index of 3d model   e.g. 'model_0_1_2'
        return: tuple of (img1, img2, binvox1, v12)
        img:[1x256x256]  binvox:[64x64x64]
        v12: ByteTensor of size1x2 e.g:[3,5]
        """
        print index
        _,model_idx,v1_idx,v2_idx = self.idx[index].split('_')

        img1_id = '{}_{}'.format(model_idx, v1_idx)
        img2_id = '{}_{}'.format(model_idx, v2_idx)

        with open(os.path.join(self.target_path, img2_id + '.binvox')) as f1:
            m = read_as_3d_array(f1)
        target2 = m.data.transpose(2, 1, 0)  ## notice here!
        #target2 = m.data

        with open(os.path.join(self.target_path, img1_id + '.binvox')) as f1:
            m = read_as_3d_array(f1)
        target1 = m.data.transpose(2, 1, 0)   ## notice here!
        #target1 = m.data

        img1 = cv2.imread(os.path.join(self.img_path, img1_id + '.jpg'), 0)
        img2 = cv2.imread(os.path.join(self.img_path, img2_id + '.jpg'), 0)


        return torch.from_numpy(img1).type(torch.FloatTensor), \
               torch.from_numpy(img2).type(torch.FloatTensor), \
               torch.FloatTensor(target1.astype(np.float32)),\
               torch.FloatTensor(target2.astype(np.float32)),(int(v1_idx),int(v2_idx))
               #torch.ByteTensor([v1,v2]).view(1,2)


    def pull_img(self, index,viewpoint):
        if index>len(self.idx) or viewpoint>12:
            print ('Warning: the index of model should <max_idx and viewpoint should <12')
        img_id = '{}_{}'.format(index, viewpoint)
        img = cv2.imread(os.path.join(self.img_path, img_id + '.jpg'), 0)
        #
        return img_id, img

    def pull_target(self, index,viewpoint):
        if index>len(self.idx) or viewpoint>12:
            print ('Warning: the index of model should <max_idx and viewpoint should <12')
        img_id = '{}_{}'.format(index, viewpoint)
        with open(os.path.join(self.target_path, img_id + '.binvox')) as f:
            m = read_as_3d_array(f)
        target = m.data
        return img_id,target

    def pull_tensor(self, index,viewpoint):
        torch.Tensor(self.pull_img(self.idx[index],viewpoint)).unsqueeze_(0)

def multi_collate(batch):
    """Custom collate fn for dealing with batches of pull_item data

        Arguments:
            batch: (tuple) img *2 .binvox*1 v12

        Return:
            A tuple containing:
                1) (tensor)*2 batch of img1,img2 stacked on their 0 dim (N,1,256,256)
                2) (tensor) annotations for .binvox data stacked on 0 dim (N,64,64,64)
                3) (tensor) tensor of viewpoint1/2 stacked on 0 dim (N,2)
        """
    target1s = []
    target2s=[]
    img1s = []
    img2s = []
    v12s = []
    for sample in batch:
        img1s.append((sample[0])[np.newaxis, :])
        img2s.append((sample[1])[np.newaxis, :])

        target1s.append((sample[2]))
        target2s.append((sample[3]))

        v12s.append((sample[4]))
    return torch.stack(img1s, 0),torch.stack(img2s, 0),\
           torch.stack(target1s, 0),torch.stack(target2s, 0), v12s








