from __future__ import print_function,division

import os
import os.path
import argparse
import time

from data_prepare.author_data import singleDataset,single_collate

from layer.unet import single_UNet
from layer.voxel_func import *
from utils.utils_rw import *

is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='single-view CNN evaluate')
parser.add_argument('--data', metavar='DIR',default='./dataset/CubeData',
                    help='path to dataset')

parser.add_argument('--data-name', default='csg', type=str, metavar='PATH',
                    help='name of dataset (default: csg)')
## this arg: name of log, data list file(.txt)

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')

parser.add_argument('-bs',  '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size ')

parser.add_argument('--resume', default='latest_model_multi.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint ')
args=parser.parse_args()

resume=args.resume
print (resume)

data_rootpath=args.data
if is_GPU:
    torch.cuda.set_device(args.gpu)


def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union

def eval_iou(pred,target):
    #print pred.size(),target.size()
    pred,target=pred.cpu().numpy().astype(np.float32)>0.5,\
                target.cpu().numpy().astype(np.float32)>0.5
    intersect=np.sum(target[pred])
    union=np.sum(pred) + np.sum(target) - intersect
    print ("output occupy sum:{}".format(np.sum(pred)))

    return intersect,intersect*1.0/union

prob=torch.nn.Softmax(dim=1)

model=single_UNet()
if is_GPU:
    model.cuda()

def evaluate():
    ## calculate the average IOU between pred and gt
    if os.path.isfile(resume):
        print ('load the checkpoint file {}'.format(resume))
        if is_GPU:
            checkoint = torch.load(resume,map_location={'cuda:0':'cuda:1'})
        else:
            checkoint = torch.load(resume, map_location=lambda storage,loc:storage)

        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        #optimizer.load_state_dict(checkoint['optim'])
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    model.eval()
    IOUs=0
    total_correct=0

    data_eval = singleDataset(data_rootpath,test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=args.batch_size, shuffle=True, collate_fn=single_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx,(imgs, targets) in enumerate(eval_loader):
        if is_GPU:
            imgs = Variable(imgs.cuda())
            targets = [Variable(anno.cuda(),requires_grad=False) for anno in targets]
        else:
            imgs = Variable(imgs)
            targets = [Variable(anno, requires_grad=False) for anno in targets]
        t1=time.time()
        outputs=model(imgs)
        outputs=prob(outputs)

        _,occupy = torch.max(outputs.data,dim=1)
        #occupy=(outputs.data[:1]>0.5)
        #print (occupy)
        occupy = occupy.view(-1,64,64,64)


        for idx,target in enumerate(targets):
            insect,iou=eval_iou(occupy[idx],target.data)
            print ('iou:',iou)
            IOUs += iou

            total_correct += insect

        t2=time.time()
        print ('in batch{} cost{}s'.format(batch_idx,t2-t1))

    #print 'correct num:{}'.format(total_correct)
    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))
    print ('the average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset))))

evaluate()

## I don't know why,but the evaluate.py iou is low(0.4),and the demo.py iou is high(0.8)
