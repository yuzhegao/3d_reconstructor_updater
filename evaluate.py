import os
import os.path
import argparse

import time
from data_prepare.bulid_data import multiDataset,multi_collate
from data_prepare.multi_loader import multiDataLoader
from layer.voxel_net2 import MulitUpdateNet
from layer.voxel_func import *
from utils.utils_rw import *

parser = argparse.ArgumentParser(description='Multi-view updater CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/CubeData',
                    help='path to dataset')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--iter', default=3, type=int, metavar='N',
                    help='number of iter in  once forward')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')


parser.add_argument('--resume', default='./model/latest_model_multi.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./model/latest_model_multi.pth)')
parser.add_argument('--single-model', default='./single_model/latest_model.pth', type=str, metavar='PATH',
                    help='path to single model (default: ./single_model/latest_model.pth)')


args=parser.parse_args()


def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union


'''
with open('26_4.binvox', 'rb') as f:
    # m1 = read_as_coord_array(f)
    m1 = read_as_3d_array(f)

with open('26_4gt.binvox', 'rb') as f1:
    # m1 = read_as_coord_array(f)
    m2 = read_as_3d_array(f1)

#print m1.data,m2.data
print IOU(m1.data,m2.data)'''

is_GPU=torch.cuda.is_available()

if is_GPU:
    torch.cuda.set_device(args.gpu)

data_rootpath=args.data
resume=args.resume

singlemodel_path=args.single_model

model=MulitUpdateNet(singlemodel_path)
if is_GPU:
    model.cuda()

def evaluate():
    ## calculate the average IOU between pred and gt
    if os.path.exists(resume):
        checkoint = torch.load(resume)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        #optimizer.load_state_dict(checkoint['optim'])
        print ('load the resume checkpoint from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    model.eval()
    IOUs=0
    total_correct=0

    data_eval = multiDataset(data_rootpath, test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=multi_collate)

    # model.SingleNet
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
    if is_GPU:
        model.SingleNet = model.SingleNet.cuda()

    for batch_idx, (img1s, img2s, targets,_, v12s) in enumerate(eval_loader):
        ## when predict,the target is target1 (img1)
        if is_GPU:
            img1s = Variable(img1s.cuda(),volatile =True)
            img2s = Variable(img2s.cuda(),volatile =True)
            targets = Variable(targets.cuda())
        else:
            img1s = Variable(img1s,volatile =True)
            img2s = Variable(img2s,volatile =True)
            targets = Variable(targets)

        t1 = time.time()
        # print v12s
        outputs = model.predict(img1s, img2s, v12s,iter=args.iter)
        t2 = time.time()
        print ('in batch{}/{} use {}s'.format(batch_idx*args.batch_size,
                                              len(eval_loader.dataset),t2-t1))

        #occupy=(outputs.data>0.5)if is_GPU else (outputs.data>0.5).type(torch.FloatTensor)
        #print occupy.sum()
        occupy = (outputs.data > 0.5)


        for idx,target in enumerate(targets):
            #correct+=(occupy[idx].eq(target.data)).sum()
            insect=(target.data[occupy[idx]]).sum()
            union=target.data.sum()+occupy[idx].sum()-insect
            iou=insect*1.0/union
            IOUs+=iou

            total_correct += insect

    #print 'correct num:{}'.format(total_correct)
    print 'the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset)))
    print 'the average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset)))

evaluate()