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

from utils.utils_rw import *

torch.set_printoptions(threshold='nan')

dims=64
dims=[dims] * 3
print dims
dims = np.atleast_2d(dims).T
print dims
dense_data = np.zeros(dims.flatten(), dtype=np.float32)
print dense_data.shape