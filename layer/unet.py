from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from utils.utils_trans import *

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    #print(classname)
    init.normal(m.weight.data,std=0.02)

def add_conv_stage(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(negative_slope=0.2)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.LeakyReLU(negative_slope=0.2)
    )

def upsample(input_fm, output_fm,use_dropout=False):
    if use_dropout:
        return nn.Sequential(
            nn.ConvTranspose2d(input_fm, output_fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_fm),
            nn.Dropout2d(0.5),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_fm, output_fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_fm),
            nn.ReLU(),
        )



###########################################################################################
## single-view predict network
class single_UNet(nn.Module):
  def __init__(self, useBN=True,net_for_training=True):
    super(single_UNet, self).__init__()
    self.net_for_train=net_for_training

    self.conv1   = add_conv_stage(3, 64, useBN=False)
    self.conv2   = add_conv_stage(64, 128, useBN=useBN)
    self.conv3   = add_conv_stage(128, 256, useBN=useBN)
    self.conv4   = add_conv_stage(256, 512, useBN=useBN)
    self.conv5   = add_conv_stage(512, 512, useBN=useBN)
    self.conv6   = add_conv_stage(512, 512, useBN=useBN)
    self.conv7   = add_conv_stage(512, 512, useBN=useBN)
    self.conv8   = add_conv_stage(512, 512, useBN=useBN)


    self.upsample87 = upsample(512, 512,use_dropout=True)
    self.upsample76 = upsample(1024, 512,use_dropout=True)
    self.upsample65 = upsample(1024, 512,use_dropout=True)
    self.upsample54 = upsample(1024, 512)
    self.upsample43 = upsample(1024, 256)
    self.upsample32 = upsample(512,  128)


    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.normal(m.weight.data, std=0.02)
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

    block7 = torch.cat((self.upsample87(conv8_out), conv7_out), dim=1)
    block6 = torch.cat((self.upsample76(block7), conv6_out), dim=1)
    block5 = torch.cat((self.upsample65(block6), conv5_out), dim=1)
    block4 = torch.cat((self.upsample54(block5), conv4_out), dim=1)
    block3 = torch.cat((self.upsample43(block4), conv3_out), dim=1)
    block2 = self.upsample32(block3)  ## here we get (N,128,64,64)

    output = block2.view(-1,2,4096*64)
    return output



