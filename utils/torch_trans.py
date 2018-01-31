

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

from data_prepare.bulid_data import multiDataset

from utils.utils_trans import *
from utils.utils_rw import *

#np.set_printoptions(threshold='nan')

#is_GPU=torch.cuda.is_available()
is_GPU=False

def dense2sparse(dense_data):
    """
    dense format(N,64,64,64) -> sparse(coord) format list of (num_nozeros,3) center_orginal

    dense_data: the pred Variable [64,64,64]
    return: Variable of sparse_data
    """
    if dense_data.data.ndimension() != 4:
        print ('when dense->sparse ,the Dimension number is Wrong!')
        print ('current data shape', dense_data.data.size())

    num_sample = dense_data.data.size()[0]
    # print (dense_data.data.size())

    sparse_list = list()
    for i in xrange(num_sample):
        ## the center is orginal,so coord(x,y,z) is float
        sparse_data = torch.nonzero(dense_data[i].data).type(torch.FloatTensor) - 32.0 + 0.5
        """
        if is_GPU:
            sparse_data = sparse_data.cuda()"""
        sparse_data=sparse_data.numpy()

        sparse_list.append(sparse_data)
    return sparse_list


def sparse2dense(sparse_list):
    """
    sparse(coord) format list of (num_nozeros,3) center_orginal -> dense format(N,64,64,64)
    param: sparse_list: the list of sparse(coord) data (3,num_nozero)

    return:dense data [N,64,64,64]
    """
    if sparse_list[0].ndimension() != 2 or sparse_list[0].size()[1] != 3:
        print ('when sparse->dense, the Dimension number is Wrong!')
        print ('current data shape:{}'.format(sparse_list[0].size()))

    num_sample = len(sparse_list)
    for i in xrange(num_sample):
        ## the center is center-of-VoxelGrid ,so the coord(x,y,z) is int
        sparse_list[i] = (sparse_list[i] + 32.0 - 0.5).type(torch.ByteTensor)

    # _,mask1=torch.max(sparse_data.data,0)
    # _,mask2=torch.min(sparse_data.data,64)
    """
    dense_data = torch.zeros(num_sample, 64, 64, 64)
    if is_GPU:
        dense_data = dense_data.cuda()
    """

    dense_list=list()

    for idx, sparse_data in enumerate(sparse_list):
        #print ('sparse_data:{}'.format(sparse_data.data.size()))  ##torch.Size([6496, 3])
        dense_data = np.zeros((64, 64, 64), dtype=np.float32)
        xyz=sparse_data.numpy().T.astype(np.int)
        valid_ix = ~np.any((xyz < 0) | (xyz >= 64), 0)
        xyz = xyz[:, valid_ix]
        print "xyz",xyz.shape
        #dense_data = np.zeros(dims.flatten(), dtype=np.float32)
        dense_data[tuple(xyz)] = 1.0
        dense_list.append(dense_data[np.newaxis,:])

    dense_data=np.concatenate(dense_list,axis=0)
    print "dense_data",dense_data.shape

    dense_data = torch.from_numpy(dense_data)
    if is_GPU:
        dense_data = dense_data.cuda()
        """
        for i in xrange(sparse_data.size()[0]):
            if sparse_data[i][0] > 0 and sparse_data[i][1] > 0 \
                    and sparse_data[i][2] > 0 and sparse_data[i][0] < 64 \
                    and sparse_data[i][1] < 64 and sparse_data[i][2] < 64:
                dense_data[idx, sparse_data[i][0], sparse_data[i][1], sparse_data[i][2]] = 1
        """

    return Variable(dense_data)  ##[N,64,64,64]
'''
a=np.random.randint(0,2,(2,64,64,64))
a=Variable(torch.from_numpy(a).type(torch.FloatTensor))
if is_GPU:
    a = Variable(torch.from_numpy(a).cuda().type(torch.FloatTensor))

b= dense2sparse(a)
print (b[0].data.size()) ##list

a=sparse2dense(b)
print a
'''

data_rootpath='../dataset/CubeData'


dataset=multiDataset(data_rootpath)
img_id, test_img = dataset.pull_img(10,2)
test_img = np.array(test_img)
#print test_img
#cv2.imshow('mat',test_img)
#cv2.waitKey(2000)
#print img_id   ## img read bingo
'''
img1,img2,target1,v12 = dataset.pull_item(10,2,7)

cv2.imshow('img1',img1.numpy())
cv2.imshow('img2',img2.numpy())
cv2.waitKey(2000)
print target1.numpy()
target_data= target1.numpy().astype(np.bool)

with open('../0_0.binvox', 'rb') as f:
    m1 = read_as_orginal_coord(f)

m1.data=target_data.transpose(2,1,0)
with open('test10_2.binvox', 'wb')as f1:
    write(m1, f1)

## test result: img->blur target(binvox)->correct
'''
v1_idx=6
v2_idx=9

_,_,target11,target12,v12_1 = dataset.pull_item(10,v1_idx,v2_idx)
_,_,target21,target22,_ = dataset.pull_item(12,v1_idx,v2_idx)

with open('../0_0.binvox', 'rb') as f:
    m1 = read_as_orginal_coord(f)

m1.data=target11.numpy().astype(np.bool)
with open('test10_6.binvox', 'wb')as f1:
    write(m1, f1)
m1.data=target21.numpy().astype(np.bool)
with open('test12_6.binvox', 'wb')as f1:
    write(m1, f1)

#target1,target2=target1[np.newaxis,:],target2[np.newaxis,:]
#print target1.size(),target2.size()
batch=list()
batch.append(target11)
batch.append(target21)
pred = torch.stack(batch,0)

#print pred.size()

if is_GPU:
    pred=pred.cuda()

pred=Variable(pred)

inv_trans_list=list()
trans_list=list()
if is_GPU:
    for trans in inv_transform_list:
        inv_trans_list.append(torch.from_numpy(trans).type(torch.FloatTensor).cuda())
    for trans in transform_list:
        trans_list.append(torch.from_numpy(trans).type(torch.FloatTensor).cuda())
else:
    for trans in inv_transform_list:
        inv_trans_list.append(torch.from_numpy(trans).type(torch.FloatTensor))
    for trans in transform_list:
        trans_list.append(torch.from_numpy(trans).type(torch.FloatTensor))


### check it!!!
#trans12 = trans_list[v2_idx].mm(inv_trans_list[v1_idx])  ## first v1_idx->9,then 9->v2_idx
#trans21 = trans_list[v1_idx].mm(inv_trans_list[v2_idx])
trans12 = transform_list[v2_idx].dot(inv_transform_list[v1_idx])
trans21 = transform_list[v1_idx].dot(inv_transform_list[v2_idx])

#trans12,trans21=Variable(trans12,requires_grad=False),Variable(trans21,requires_grad=False)

print ('trans12')
print trans12
#print trans21 ## transform metrix : correct

preds=dense2sparse(pred>0.5)  ## list of (num_nonzero,3)
#print 'before transform in torch',preds[0]

preds_trans=list()
for pred in preds:
    print ('pred data shape')
    print(pred.shape)  ##(177609L, 3L) [n,zyx]

    print 'before transform in torch',pred.T ##[zyx,n]
    #pred = trans12.mm(pred.view(3,-1))
    pred = trans12.dot(pred.T[::-1,:]) ##[xyz,n]
    pred =pred[::-1,:].T ## [n,zyx]
    pred=torch.from_numpy(pred.astype(np.float32)).type(torch.FloatTensor)

    preds_trans.append(pred)
print 'after transform in torch',preds_trans[0]

preds=sparse2dense(preds_trans)
print 'check3',preds.data.size()


binvox_data1=(preds.data[0]>0.5).numpy().transpose(2,1,0)
binvox_data2=(preds.data[1]>0.5).numpy().transpose(2,1,0)

with open('../0_0.binvox', 'rb') as f:
    m1 = read_as_orginal_coord(f)

m1.data=binvox_data1
print 'm1.data',m1.data.shape
with open('test10_9.binvox', 'wb')as f1:
    write(m1, f1)

m1.data=binvox_data2
with open('test12_9.binvox', 'wb')as f1:
    write(m1, f1)


'''

dataset=singleDataset(data_rootpath)
img_id, test_img = dataset.pull_img(10)
test_img = np.array(test_img)
#test_img=cv2.imread('../dataset/CubeData/img/10_2.jpg')
print test_img
cv2.imshow('mat',test_img)
cv2.waitKey(5000)
print img_id
'''



