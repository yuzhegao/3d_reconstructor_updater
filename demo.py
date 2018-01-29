"""
use model.predict function to predict 3D model (with 2~3 iterate)
"""

import os.path
import time
import random

from utils.utils_rw import *

import numpy as np
from data_prepare.bulid_data import multiDataset
from layer.voxel_net2 import MulitUpdateNet
from layer.voxel_func import *

is_GPU=torch.cuda.is_available()

if is_GPU:
    torch.cuda.set_device(3)

def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union

def eval_model(model_name,demo_path):
    with open('./dataset/CubeData/binvox/'+model_name+'.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_3d_array(f)

    with open(demo_path + model_name+'.binvox', 'rb') as f1:
        # m1 = read_as_coord_array(f)
        m2 = read_as_3d_array(f1)

    # print m1.data,m2.data
    print "IOU={:.3f}".format(IOU(m1.data, m2.data))
    print ' '

#np.set_printoptions(threshold='nan')
resume='./model/latest_model_multi_2nd.pth'
singlemodel_path='./single_model/t_latest_model.pth'

#model=singleNet()
model=MulitUpdateNet(singlemodel_path)

if os.path.exists(resume):
    if is_GPU:
        model.load_state_dict(torch.load(resume)['model'])
    else:
        model.load_state_dict(torch.load(resume,map_location=lambda storage, loc: storage)['model'])
    print  'load trained model success'

data_rootpath='./dataset/CubeData'
dataset=multiDataset(data_rootpath)

model.eval()


def generate_binvox(num1,num2,demo_path='./demo2/'):
    for i in xrange(num1, num2):
        v1 = random.randint(0, 7)
        v2 = random.randint(0, 7)

        #v2=v1
        while(v1==v2):
            v2 = random.randint(0, 7)

        img1_id, test_img1 = dataset.pull_img(i, v1)
        img2_id, test_img2 = dataset.pull_img(i, v2)
        print img1_id, img2_id

        test_img1 = test_img1[np.newaxis, :][np.newaxis, :]
        test_img2 = test_img2[np.newaxis, :][np.newaxis, :]

        # print test_img1.shape
        img1 = torch.from_numpy(test_img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(test_img2).type(torch.FloatTensor)

        img1 = Variable(img1,volatile =True)
        img2 = Variable(img2,volatile =True)

        init = time.time()
        up = model.predict(img1,img2,[(v1,v2)],2)  ##iterate 2 times
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
        with open(demo_path + img1_id + '.binvox', 'wb')as f1:
            write(m1, f1)
        eval_model(img1_id,demo_path)

path='./demo_second/demo7_7/'
generate_binvox(210,230,demo_path=path)
generate_binvox(1210,1230,demo_path=path)
generate_binvox(2210,2230,demo_path=path)

