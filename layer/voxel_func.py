from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class VoxelLoss(nn.Module):
    def __init__(self):
        super(VoxelLoss,self).__init__()

    def forward(self,outputs,targets):
        '''
        :param outputs:  predict output [batchsize,64,64,64]
        :param targets:  target voxel grid ,a list of [64,64,64]float size=batchsize
        '''
        loss=torch.zeros(1)
        zero_mat=Variable(torch.zeros(64,64,64))

        if torch.cuda.is_available():
            loss=Variable(loss.cuda())
        else:
            loss=Variable(loss)
        for batch_idx,target in enumerate(targets):

            loss_mat = torch.abs(target - outputs[batch_idx, :, :, :])       
            loss += torch.sum(loss_mat)
        return loss/len(targets)

class VoxelL1(nn.Module):
    def __init__(self):
        super(VoxelL1,self).__init__()

    def forward(self,outputs,targets):
        '''
        :param outputs:  predict output [batchsize,64,64,64]
        :param targets:  target voxel grid ,[batchsize,64,64,64] float
        '''
        loss=torch.zeros(1)

        if torch.cuda.is_available():
            loss=Variable(loss.cuda())
        else:
            loss=Variable(loss)
        '''
        for batch_idx,target in enumerate(targets):

            loss_mat = torch.abs(target - outputs[batch_idx, :, :, :])       
            loss += torch.sum(loss_mat)
        '''
        loss = torch.sum( torch.abs(targets-outputs) )
        return loss/len(targets)

class CrossEntropy_loss(nn.Module):
    def __init__(self):
        super(CrossEntropy_loss,self).__init__()

    def forward(self,outputs,targets,gamma=0.5):
        loss=torch.zeros(1)
        if torch.cuda.is_available():
            loss=Variable(loss.cuda())
        else:
            loss=Variable(loss)

        #targets=targets*3-1.0  ## clamp targets value to {-1,2}
        #print (targets)
        #print (torch.sum(-gamma*targets*torch.log(outputs)))
        #loss=torch.sum(-gamma*targets*torch.log(outputs)-(1-gamma)*(1-targets)*torch.log(1-outputs))
        loss=-1.0*torch.sum(-gamma*targets*torch.log(outputs+10**-9))\
             -1.0*torch.sum((1-gamma)*(1-targets)*torch.log(1-outputs+10**-9))
        return loss/len(targets)


