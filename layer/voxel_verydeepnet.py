from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

#from voxel_net2 import add_conv_stage


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    #print(classname)
    init.xavier_uniform(m.weight.data)
    #init.xavier_uniform(m.bias.data)

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(2)
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

class singleNet_verydeep(nn.Module):
  def __init__(self, useBN=True):
    super(singleNet_verydeep, self).__init__()

    self.conv1   = add_conv_stage(1, 32, useBN=useBN)
    self.conv2   = add_conv_stage(32, 64, useBN=useBN)
    self.conv3   = add_conv_stage(64, 128, useBN=useBN)
    self.conv4   = add_conv_stage(128, 256, useBN=useBN)
    self.conv5   = add_conv_stage(256, 512, useBN=useBN)
    self.conv6   = add_conv_stage(512, 1024, useBN=useBN)
    self.conv7   = add_conv_stage(1024, 2048, useBN=useBN)
    self.conv8   = add_conv_stage(2048, 4096, useBN=useBN)

    self.upsample87 = upsample(4096, 2048)
    self.upsample76 = upsample(4096, 1024)
    self.upsample65 = upsample(2048, 512)
    self.upsample54 = upsample(1024, 256)
    self.upsample43 = upsample(512, 128)
    self.upsample32 = upsample(256,  64)

    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()


  def forward(self, x):
    conv1_out = self.conv1(x)
    conv2_out = self.conv2(conv1_out)
    conv3_out = self.conv3(conv2_out)
    conv4_out = self.conv4(conv3_out)
    conv5_out = self.conv5(conv4_out)
    conv6_out = self.conv6(conv5_out)
    conv7_out = self.conv7(conv6_out)
    conv8_out = self.conv8(conv7_out)


    conv7m_out = torch.cat((self.upsample87(conv8_out), conv7_out), 1)
    conv6m_out = torch.cat((self.upsample76(conv7m_out), conv6_out), 1)
    conv5m_out = torch.cat((self.upsample65(conv6m_out), conv5_out), 1)
    conv4m_out = torch.cat((self.upsample54(conv5m_out), conv4_out), 1)
    conv3m_out = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv2m_out = self.upsample32(conv3m_out)


    #outputs=F.sigmoid(self.conv_last(conv2m_out))
    ## find if the last layer is conv,the loss drop rapidly and get all-zero result
    #outputs = torch.clamp(F.sigmoid(self.max_pool(conv2m_out)),min=0.1,max=1.0)
    #print (conv1_out.data.size())
    #outputs = F.sigmoid(self.max_pool(conv2m_out))
    return conv2m_out


