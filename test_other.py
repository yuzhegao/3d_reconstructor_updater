from __future__ import print_function,division

import os
import os.path
import sys
import time
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from utils.utils_rw import *

import cv2
import numpy as np
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_func import *

'''
img=cv2.imread('/home/gaoyuzhe/PycharmProjects/3Dbuilder/dataset/img/2_6.jpg')
cv2.imshow('img',img)
cv2.waitKey(2000)
'''
'''
np.set_printoptions(threshold='nan')

data_rootpath='/home/gaoyuzhe/PycharmProjects/3Dbuilder/dataset'
dataset=singleDataset(data_rootpath)

test_target=None
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=single_collate)
for batch_idx,(imgs,targets) in enumerate(data_loader):
    print batch_idx
    print np.sum(targets[1].numpy())
    test_target=targets[1]
    break

test_target=test_target.numpy()
test_target=(test_target>0.5)

with open('/home/gaoyuzhe/PycharmProjects/3Dbuilder/dataset/binvox/2_6.binvox','rb') as f:
    m1=read_as_coord_array(f)
    print m1.data.shape[1]
with open('test.binvox','wb') as f1:
    m1.data=test_target
    write(m1,f1)


'''
a=torch.randn(2,64,64,64)
print (a.transpose(3,1))
