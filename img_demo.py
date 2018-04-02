"""
use the trained model to predict 3D model
from an alterative given image

for example:
single demo   python img_demo.py --img1 './22_4.jpg' --gt './22_4.binvox'
multi demo    python img_demo.py --img1 './22_4.jpg' --img2 './22_0.jpg' --v1 4 --v2 0 --gt './22_0.binvox'
"""
from __future__ import print_function,division
import os.path
import cv2
import time

import random

from utils.utils_rw import *
from utils.utils_trans import inv_transform_list

import numpy as np
import argparse
import torchvision
#from layer.voxel_deepernet import singleNet_deeper,MulitUpdateNet_deeper
from layer.voxel_verydeepnet import singleNet_verydeep,MulitUpdateNet_verydeep
from layer.voxel_func import *

parser = argparse.ArgumentParser(description='3d_predict CNN demo')
parser.add_argument('--img1', metavar='DIR',default='./22_0.jpg',
                    help='path to the line drawing image')
parser.add_argument('--img2', metavar='DIR',default=None,
                    help='path to the line drawing image2')

parser.add_argument('--v1', default=0, type=int, metavar='N',
                    help='the viewpoint index  of img1')
parser.add_argument('--v2', default=None, type=int, metavar='N',
                    help='the viewpoint index  of img2')

parser.add_argument('--gt', default='22_0.binvox', type=str, metavar='N',
                    help='path to ground truth voxel grid (in multi_demo,the gt of viewpoint 2)')

parser.add_argument('--resume-single', default='csg_single_train_ce_server_87.pth', type=str, metavar='PATH',
                    help='name of latest checkpoint file (single model)')
parser.add_argument('--resume-multi', default='csg_multi_train_ce_11.pth', type=str, metavar='PATH',
                    help='name of latest checkpoint file (multi model)')
args=parser.parse_args()

is_multi=False

def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    print ('output occupy voxel:{}'.format(np.sum(model2)))
    return insect*1.0/union

def eval_model():
    #print(model_name)
    with open('result.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_3d_array(f)

    with open(args.gt, 'rb') as f1:
        # m1 = read_as_coord_array(f)
        m2 = read_as_3d_array(f1)

    # print m1.data,m2.data
    print ('iou:',IOU(m1.data, m2.data))

def read_img(img_path):
    img_nptensor = cv2.imread(img_path)
    if img_nptensor is None:
        print ('cannot find img {}'.format(img_path))
        exit()
    transf=torchvision.transforms.ToTensor()

    img=transf(img_nptensor)
    return img




if args.img1 is None:
    print ('Warining! img1 is NONE')
    exit()

if args.img2 is not None and args.v2 is not None :
    print ('use multi-view network to test')
    is_multi=True
else:
    print ('use single-view network to test')




def demo_single(img1_path):
    resume_single = './model/' + args.resume_single
    model=singleNet_verydeep()

    if os.path.exists(resume_single):
        model.load_state_dict(torch.load(resume_single, map_location=lambda storage, loc: storage)['model'])
        print('load trained model {} success'.format(resume_single))
    else:
        print("Warning! no resume file to load\n")

    img1_tensor=read_img(img1_path)
    img1_tensor=torch.unsqueeze(img1_tensor,0)

    img = Variable(img1_tensor)
    init = time.time()
    up = model(img)
    end = time.time()

    result = up.data[0, :, :, :].numpy()
    # print (result)
    # print ("each forward use:{}s".format(end - init))
    result = (result >= 0.5)

    with open('0_0.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_orginal_coord(f)

    m1.data = result.transpose(2, 1, 0)
    with open('result.binvox', 'wb')as f1:
        write(m1, f1)  ## write the result 64x64x64 data to .binvox file
    eval_model()



def multi_train(img1_path, img2_path, v1, v2):
    resume='./model/' + args.resume_multi
    model=MulitUpdateNet_verydeep()

    if os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=lambda storage, loc: storage)['model'])
        print('load trained model {} success'.format(resume))
    else:
        print ('Warning! no resume file to load\n\n')

    resume_single = './model/' + args.resume_single
    if os.path.exists(resume_single):
        checkoint = torch.load(resume_single, map_location=lambda storage, loc: storage)
        model.SingleNet.load = model.SingleNet.load_state_dict(checkoint['model'])

        print('load trained single model {} success'.format(resume_single))
    else:
        print("Warning! no single resume file to load\n")


    img1_tensor=read_img(img1_path)
    img2_tensor=read_img(img2_path)

    img1 = Variable(img1_tensor)
    img2 = Variable(img2_tensor)

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
    with open('result.binvox', 'wb')as f1:
        write(m1, f1)
    eval_model()


if is_multi:
    multi_train(args.img1, args.img2, args.v1, args.v2)
else:
    demo_single(args.img1)









