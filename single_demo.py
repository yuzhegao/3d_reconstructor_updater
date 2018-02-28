from __future__ import print_function,division

import os.path

import time
import argparse
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_net2 import singleNet
from layer.voxel_func import *
from utils.utils_rw import *

parser = argparse.ArgumentParser(description='single-view CNN demo')
parser.add_argument('--data', metavar='DIR',default='./dataset/CubeData',
                    help='path to dataset')

parser.add_argument('--data-name', default='csg', type=str, metavar='PATH',
                    help='name of dataset (default: csg)')
## this arg: name of log, data list file(.txt)

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')

parser.add_argument('-bs',  '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')

parser.add_argument('--resume', default='latest_model_multi.pth', type=str, metavar='PATH',
                    help='name of latest checkpoint file (default: latest_model_multi.pth)')
args=parser.parse_args()


def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    print ("insect:{} union:{}".format(insect,union))
    return insect*1.0/union

def eval_model(model_name):
    with open(args.data+'/binvox/'+model_name+'.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_3d_array(f)

    with open('./demo/'+model_name+'.binvox', 'rb') as f1:
        # m1 = read_as_coord_array(f)
        m2 = read_as_3d_array(f1)

    # print m1.data,m2.data
    print ('iou:',IOU(m1.data, m2.data))



np.set_printoptions(threshold='nan')
resume='./model/'+args.resume

model=singleNet()
##model.eval()  this is the problem!!!

is_GPU=torch.cuda.is_available()
if is_GPU:
    torch.cuda.set_device(args.gpu)

if os.path.exists(resume):
    print('load the checkpoint file {}'.format(resume))
    if is_GPU:
        model.load_state_dict(torch.load(resume)['model'])
    else:
        model.load_state_dict(torch.load(resume,map_location=lambda storage, loc: storage)['model'])

    print ('load trained model success')
else:
    print ("Warning! no resume file to load\n")

data_rootpath=args.data
dataset=singleDataset(data_rootpath,data_name=args.data_name,test=True)


## very strange:  for i in xrange(10010,10013): use cube_single_train.pth trained in server,get a very high iou(0.8)
## but in evaluate_single, the iou is low(0.4)
for i in xrange(20,23):
    img_id, test_img = dataset.pull_img(i)
    # test_img = np.array(test_img).astype(np.float32)
    # cv2.imshow('mat',test_img)
    # cv2.waitKey(1000)
    print (img_id)

    test_img = test_img[np.newaxis, :]
    test_img = test_img[np.newaxis, :]

    #print test_img.shape
    img = torch.from_numpy(test_img).type(torch.FloatTensor)
    img = Variable(img)
    init = time.time()
    up = model(img)
    end = time.time()

    result = up.data[0, :, :, :].numpy()
    #print (result)
    print ("each forward use:{}s".format(end - init))
    result = (result >= 0.5)

    with open('0_0.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_orginal_coord(f)

    m1.data = result.transpose(2,1,0)
    with open('./demo/'+img_id+'.binvox', 'wb')as f1:
        write(m1, f1)
    eval_model(img_id)



