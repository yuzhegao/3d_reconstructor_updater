"""
test if the transform in torch(GPU) is correct
"""

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
from utils.torch_trans import sparse2dense,dense2sparse

################################################
print ('\n \n')
print 'trans6_9'
print inv_transform_list[6]

with open('10_6.binvox', 'rb') as f:
    # m1 = read_as_coord_array(f)
    m1 = read_as_orginal_coord(f)
    print 'before transform ',m1.data
    print m1.data.shape

    #transform = trans6_9()
    transform = inv_transform_list[6]
    trans_coord = transform.dot(m1.data)
    print 'after transform'
    print trans_coord
    print trans_coord.shape

trasform_voxel = orgin_sparse_to_dense(trans_coord, 64)
#print trasform_voxel

m1.data = trasform_voxel
with open('10_t9.binvox', 'wb')as f1:
    write(m1, f1)

###############################################
with open('10_t9.binvox', 'rb') as f:
    # m1 = read_as_coord_array(f)
    m1 = read_as_orginal_coord(f)

    transform = transform_list[6]
    trans_coord = transform.dot(m1.data)
    #print trans_coord

trasform_voxel = orgin_sparse_to_dense(trans_coord, 64)
#print trasform_voxel

m1.data = trasform_voxel
with open('10_t6.binvox', 'wb')as f1:
    write(m1, f1)


#print 'trans6_9'
#print inv_transform_list[6]
#print 'trans9_6'
#print transform_list[6]

