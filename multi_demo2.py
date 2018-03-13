"""
use model.forward function to predict 3D model (without iterate)
"""

from __future__ import print_function,division

import os.path
import time
import random

from utils.utils_rw import *

import numpy as np
from data_prepare.bulid_data import multiDataset
#from layer.voxel_net2 import MulitUpdateNet
from  layer.voxel_deepernet import MulitUpdateNet_deeper
from layer.voxel_func import *

is_GPU=torch.cuda.is_available()

if is_GPU:
    torch.cuda.set_device(3)

def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union

def eval_model(model_name,demo_path):
    with open('./dataset/CsgData/binvox/'+model_name+'.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_3d_array(f)

    with open(demo_path + model_name+'.binvox', 'rb') as f1:
        # m1 = read_as_coord_array(f)
        m2 = read_as_3d_array(f1)

    # print m1.data,m2.data
    print ("IOU={:.3f}".format(IOU(m1.data, m2.data)))
    print (' ')

#np.set_printoptions(threshold='nan')
#resume='./model/latest_model_multi.pth'

resume='./model/csg_multi_train_ce_11.pth'
singlemodel_path='./single_model/csg_single_train_ce_server_87.pth'

#model=singleNet()
model=MulitUpdateNet_deeper()

if os.path.exists(resume):
    if is_GPU:
        model.load_state_dict(torch.load(resume)['model'])
    else:
        model.load_state_dict(torch.load(resume,map_location=lambda storage, loc: storage)['model'])
    print  ('load trained model success')

data_rootpath='./dataset/CsgData'
dataset=multiDataset(data_rootpath,'csg',test=True)


def generate_binvox(num1,num2,demo_path='./demo2/'):
    if os.path.exists(singlemodel_path):
        t1 = time.time()
        if is_GPU:
            checkoint = torch.load(singlemodel_path)
        else:
            checkoint = torch.load(singlemodel_path, map_location=lambda storage, loc: storage)
        ## notice :here we should map the order of GPU
        ## when the SingleModel is trained in GPU-1,we should map the model to GPU-0

        model.SingleNet.load = model.SingleNet.load_state_dict(checkoint['model'])
        t2 = time.time()
        print ('singleNetwork load resume model from epoch{} use {}s'.format(checkoint['epoch'], t2 - t1))
    else:
        print('Warning: no single model to load!!!\n\n')

    for i in xrange(num1, num2):
        v1 = random.randint(0, 7)
        v2 = random.randint(0, 7)
        #v2 = random.randint(8, 12)

        #v2=v1
        while(v1==v2):
            v2 = random.randint(0, 7)

        img1_id, test_img1 = dataset.pull_img(i, v1)
        img2_id, test_img2 = dataset.pull_img(i, v2)
        print (img1_id, img2_id)

        test_img1 = test_img1[np.newaxis, :][np.newaxis, :]
        test_img2 = test_img2[np.newaxis, :][np.newaxis, :]

        # print test_img1.shape
        img1 = torch.from_numpy(test_img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(test_img2).type(torch.FloatTensor)

        img1 = Variable(img1)
        img2 = Variable(img2)

        init = time.time()
        # up = model.predict(img1,img2,[(v1,v2)],2)
        up = model(img1, img2, [(v1, v2)], None)  ##look here! just use model.forward fn
        end = time.time()

        result = up.data[0, :, :, :].numpy()
        # print result
        # print "each forward use:{}s".format(end - init)
        result = result >= 0.5

        with open('0_0.binvox', 'rb') as f:
            # m1 = read_as_coord_array(f)
            m1 = read_as_orginal_coord(f)

        m1.data = result.transpose(2, 1, 0)
        # m1.data = result
        with open(demo_path + img2_id + '.binvox', 'wb')as f1:
            write(m1, f1)
        eval_model(img2_id,demo_path)

parh='./demo2_second/demo7_7/'

for i in xrange(18):
    generate_binvox(i*100+95, i*100+99, demo_path=parh)


