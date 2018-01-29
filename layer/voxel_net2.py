import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import numpy
from utils.utils_trans import *
from utils.utils_eval import singlenet_eval


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )

def upsample(input_fm, output_fm):
  return nn.Sequential(
    nn.ConvTranspose2d(input_fm, output_fm, 2, 2, 0, bias=False),
    nn.ReLU()
  )

class singleNet(nn.Module):
  def __init__(self, useBN=True):
    super(singleNet, self).__init__()

    self.conv1   = add_conv_stage(1, 32, useBN=useBN)
    self.conv2   = add_conv_stage(32, 64, useBN=useBN)
    self.conv3   = add_conv_stage(64, 128, useBN=useBN)
    self.conv4   = add_conv_stage(128, 256, useBN=useBN)
    self.conv5   = add_conv_stage(256, 512, useBN=useBN)

    self.conv4m = add_conv_stage(512, 256, useBN=useBN)
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)

    self.max_pool = nn.MaxPool2d(2)

    self.upsample54 = upsample(512, 256)
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)

    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()


  def forward(self, x):
    conv1_out = self.conv1(x)
    conv2_out = self.conv2(self.max_pool(conv1_out))
    conv3_out = self.conv3(self.max_pool(conv2_out))
    conv4_out = self.conv4(self.max_pool(conv3_out))
    conv5_out = self.conv5(self.max_pool(conv4_out))

    conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv4m_out = self.conv4m(conv5m_out)

    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv3m_out = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out = self.conv2m(conv3m_out_)

    outputs=F.sigmoid(self.max_pool(conv2m_out))
    return outputs


class MulitUpdateNet(nn.Module):
  def __init__(self,SingleModel='./single_model/t_latest_model.pth', useBN=True,use_GPU=True):
    super(MulitUpdateNet, self).__init__()

    self.GPU = use_GPU and torch.cuda.is_available()
    print ('use GPU or not:{}'.format(self.GPU))

    self.SingleNet=singleNet()
    self.singlePath=SingleModel

    if self.GPU:
      self.SingleNet=self.SingleNet.cuda()

    for param in self.SingleNet.parameters():
      param.requires_grad = False


    self.inv_trans=list()
    self.trans=list()
    if self.GPU:
      for trans in inv_transform_list:
        self.inv_trans.append(torch.from_numpy(trans).type(torch.FloatTensor).cuda())
      for trans in transform_list:
        self.trans.append(torch.from_numpy(trans).type(torch.FloatTensor).cuda())
    else:
        for trans in inv_transform_list:
          self.inv_trans.append(torch.from_numpy(trans).type(torch.FloatTensor))
        for trans in transform_list:
          self.trans.append(torch.from_numpy(trans).type(torch.FloatTensor))
    #print len(self.inv_trans),len(self.trans)

    self.conv1 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3 = add_conv_stage(128, 128, useBN=useBN)
    self.conv4 = add_conv_stage(128, 256, useBN=useBN)
    self.conv5 = add_conv_stage(256, 512, useBN=useBN)

    self.conv4m = add_conv_stage(512, 256, useBN=useBN)
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128, 64, useBN=useBN)

    self.max_pool = nn.MaxPool2d(2)

    self.upsample54 = upsample(512, 256)
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128, 64)

    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()

  ## define the trans fn in torch format
  def dense2sparse(self,dense_data):
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
      sparse_data = sparse_data.numpy()

      sparse_list.append(sparse_data)
    return sparse_list

  def sparse2dense(self,sparse_list):
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

    dense_list = list()

    for idx, sparse_data in enumerate(sparse_list):
      # print ('sparse_data:{}'.format(sparse_data.data.size()))  ##torch.Size([6496, 3])
      dense_data = np.zeros((64, 64, 64), dtype=np.float32)
      xyz = sparse_data.numpy().T.astype(np.int)
      valid_ix = ~np.any((xyz < 0) | (xyz >= 64), 0)
      xyz = xyz[:, valid_ix]
      #print "xyz", xyz.shape
      # dense_data = np.zeros(dims.flatten(), dtype=np.float32)
      dense_data[tuple(xyz)] = 1.0
      dense_list.append(dense_data[np.newaxis, :])

    dense_data = np.concatenate(dense_list, axis=0)
    #print "dense_data", dense_data.shape

    dense_data = torch.from_numpy(dense_data)
    if self.GPU:
      dense_data = dense_data.cuda()
      """
      for i in xrange(sparse_data.size()[0]):
          if sparse_data[i][0] > 0 and sparse_data[i][1] > 0 \
                  and sparse_data[i][2] > 0 and sparse_data[i][0] < 64 \
                  and sparse_data[i][1] < 64 and sparse_data[i][2] < 64:
              dense_data[idx, sparse_data[i][0], sparse_data[i][1], sparse_data[i][2]] = 1
      """

    return Variable(dense_data)  ##[N,64,64,64]



  ## when training, just iterate once
  ## after training, we can iterate 5 times to refine the output

  def forward(self,x1,x2,v12s,target1s=None):
    """

    :param x1: img1 (N,1,256,256)
    :param x2: img2 (N,1,256,256)
    :param v12s: list of v12 e.g. [(1,2),(3,4)......]  len(v12s)=N
    :param target1s: just for evaluate the self.SingleNet ,[N,64,64,64]

    """


    self.SingleNet.eval()
    preds = self.SingleNet(x1)  ##[N,64,64,64]
    preds = preds > 0.5
    #print preds.data.sum()
    #print preds ##[torch.FloatTensor of size Nx64x64x64]
    #singlenet_eval(preds.data,target1s)

    preds = self.dense2sparse(preds)  ## list of (num_nonzero,3)

    preds_trans = list()
    #for pred in preds:
    for i in xrange(len(preds)):

      #print ('pred data shape')
      #print(pred.shape)  ##(177609L, 3L) [n,zyx]

      #print 'before transform in torch', pred.T  ##[zyx,n]
      # pred = trans12.mm(pred.view(3,-1))
      trans12=transform_list[v12s[i][1]].dot(inv_transform_list[v12s[i][0]])
      pred = trans12.dot(preds[i].T[::-1, :])  ##[xyz,n]
      pred = pred[::-1, :].T  ## [n,zyx]
      pred = torch.from_numpy(pred.astype(np.float32)).type(torch.FloatTensor)

      preds_trans.append(pred)

    preds = self.sparse2dense(preds_trans)  ## [N,64,64,64]


    conv1_out = self.conv1(x2)  ##[N,32,256,256]
    conv2_out = self.conv2(self.max_pool(conv1_out))  ##[N,64,128,128]

    conv2_out_cat = torch.cat((self.max_pool(conv2_out), preds), 1)

    conv3_out = self.conv3(conv2_out_cat)
    conv4_out = self.conv4(self.max_pool(conv3_out))
    conv5_out = self.conv5(self.max_pool(conv4_out))

    conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv4m_out = self.conv4m(conv5m_out)

    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv3m_out = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out = self.conv2m(conv3m_out_)

    preds = F.sigmoid(self.max_pool(conv2m_out))

    return preds

  def predict(self,x1,x2,v12s,iter=5):
    ## x1: view1 img for initial prediction
    ## x2: view2 img for multi-view update
    self.SingleNet.eval()
    preds=self.SingleNet(x1) ##[N,64,64,64]
    #preds=preds.transpose(3,1)
    #print 'singlenet output',preds.data.is_cuda

    for i in xrange(iter):
      ## img2 + pred1 ->CNN =pred2
      preds=self.dense2sparse(preds>0.5)  ## list of (num_nonzero,3)

      preds_trans = list()
      for i in xrange(len(preds)):

        trans12 = transform_list[v12s[i][1]].dot(inv_transform_list[v12s[i][0]])
        pred = trans12.dot(preds[i].T[::-1, :])  ##[xyz,n]
        pred = pred[::-1, :].T  ## [n,zyx]
        pred = torch.from_numpy(pred.astype(np.float32)).type(torch.FloatTensor)

        preds_trans.append(pred)

      preds=self.sparse2dense(preds_trans) ## [N,64,64,64]   ->[N,C,H,W]


      conv1_out = self.conv1(x2) ##[N,32,256,256]
      conv2_out = self.conv2(self.max_pool(conv1_out))  ##[N,64,128,128]

      conv2_out_cat= torch.cat((self.max_pool(conv2_out),preds),1)

      conv3_out = self.conv3(conv2_out_cat)
      conv4_out = self.conv4(self.max_pool(conv3_out))
      conv5_out = self.conv5(self.max_pool(conv4_out))

      conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
      conv4m_out = self.conv4m(conv5m_out)

      conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
      conv3m_out = self.conv3m(conv4m_out_)

      conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
      conv2m_out = self.conv2m(conv3m_out_)

      preds = F.sigmoid(self.max_pool(conv2m_out))

      ## img1 + pred2 ->CNN =pred3
      preds = self.dense2sparse(preds > 0.5)  ## list of (1,num_nonzero,3)

      preds_trans = list()
      for i in xrange(len(preds)):

        trans21 = transform_list[v12s[i][0]].dot(inv_transform_list[v12s[i][1]])
        pred = trans21.dot(preds[i].T[::-1, :])  ##[xyz,n]
        pred = pred[::-1, :].T  ## [n,zyx]
        pred = torch.from_numpy(pred.astype(np.float32)).type(torch.FloatTensor)

        preds_trans.append(pred)

      preds = self.sparse2dense(preds_trans)  ## [N,64,64,64]

      conv1_out = self.conv1(x2)  ##[N,32,256,256]
      conv2_out = self.conv2(self.max_pool(conv1_out))  ##[N,64,128,128]

      conv2_out_cat = torch.cat((self.max_pool(conv2_out), preds), 1)

      conv3_out = self.conv3(conv2_out_cat)
      conv4_out = self.conv4(self.max_pool(conv3_out))
      conv5_out = self.conv5(self.max_pool(conv4_out))

      conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
      conv4m_out = self.conv4m(conv5m_out)

      conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
      conv3m_out = self.conv3m(conv4m_out_)

      conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
      conv2m_out = self.conv2m(conv3m_out_)

      preds = F.sigmoid(self.max_pool(conv2m_out))

    return preds
