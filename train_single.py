from __future__ import print_function,division
import os
import os.path
import numpy as np
import torch

import time
import argparse
from data_prepare.bulid_data import singleDataset,single_collate
#from layer.voxel_net2 import singleNet
#from layer.voxel_deepernet import singleNet_deeper,weights_init
#from layer.voxel_verydeepnet import singleNet_verydeep,weights_init

from layer.unet import single_UNet,weights_init,softmax_loss
#from layer.voxel_func import *
from torch.autograd import Variable


is_GPU=torch.cuda.is_available()
#torch.set_printoptions(threshold=float('Inf'))

parser = argparse.ArgumentParser(description='Single-view reconstruct CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/A_VaseData',
                    help='path to dataset')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-step', default=50, type=int, metavar='N',
                    help='number of batch num to write log')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.7, type=float,
                    metavar='GM,', help='param of cross entropy loss')

parser.add_argument('--data-name', default='A_vase', type=str, metavar='PATH',
                    help='name of dataset (default: csg)')
## this arg: name of log, data list file(.txt)

parser.add_argument('--resume', default='csg_single_model.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: csg_single_model.pth)')

args=parser.parse_args()


data_rootpath=args.data
resume='./model/'+args.resume
## args.resume: just the name of checkpoint file

logfile=args.data_name+'_single_train.txt'
print ('logfile name:{}'.format(logfile))

if is_GPU:
    torch.cuda.set_device(args.gpu)

dataset=singleDataset(data_rootpath,data_name=args.data_name)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=single_collate)

model=single_UNet()
if is_GPU:
    model.cuda()

critenrion=softmax_loss()

optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.5,0.999))


def save_checkpoint(epoch,model,optimizer,is_epoch=False):
    torch.save({
        'model':model.state_dict(),
        'optim':optimizer.state_dict(),
        'epoch':epoch,
    },'./model/'+args.resume)
    if is_epoch:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
        }, './model_epoch/'+args.resume)
    #        print ("save model of epoch{}".format(epoch))


def log(filename,epoch,batch,loss):
    f1=open(filename,'a')
    if epoch == 0 and batch == 100:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    f1.write('\nin epoch{} batch{} loss={}'.format(epoch,batch,loss))


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

def evaluate():
    if os.path.isfile(resume):
        print ('load the checkpoint file {}'.format(resume))
        if is_GPU:
            checkoint = torch.load(resume)
        else:
            checkoint = torch.load(resume, map_location=lambda storage,loc:storage)

        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        #optimizer.load_state_dict(checkoint['optim'])
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("Warning! no resume checkpoint to load")

    model.eval()
    IOUs=0
    total_correct=0

    data_eval = singleDataset(data_rootpath,data_name=args.data_name,test=True)
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
        outputs=model(imgs)

        occupy = (outputs.data[:,1] > 0.5)  ## ByteTensor


        for idx,target in enumerate(targets):

            insect,iou=eval_iou(occupy[idx],target.data)
            IOUs += iou

            total_correct += insect


    #print 'correct num:{}'.format(total_correct)
    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))
    print ('the average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset))))
    with open(logfile,'a') as f:
        f.write('the evaluate average iou:{}\n'.format(IOUs*1.0/(len(eval_loader.dataset))))



def train():
    model.train()
    model.apply(weights_init)

    start_epoch = args.start_epoch
    num_epochs = args.epochs

    if os.path.isfile(resume):
        checkoint = torch.load(resume,map_location={'cuda:0':'cuda:3'})
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])

        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()
        for batch_idx, (imgs, targets) in enumerate(data_loader):
            if is_GPU:
                imgs = Variable(imgs.cuda())
                #targets = [Variable(anno.cuda(), requires_grad=False) for anno in targets]
                targets=Variable(targets.cuda())
            else:
                imgs = Variable(imgs)
                #targets = [Variable(anno, requires_grad=False) for anno in targets]
                targets=Variable(targets)

            t1=time.time()
            outputs = model(imgs)
            #print (outputs.data.size())

            #loss = critenrion(outputs, targets,gamma=0.5)
            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2=time.time()
            print ("in batch:{} loss={} use time:{}s".format(batch_idx, loss.data[0],t2-t1))
            #evaluate()
            if batch_idx%args.log_step==0 and batch_idx!=0:
                save_checkpoint(epoch, model, optimizer)
                log(logfile, epoch, batch_idx, loss.data[0])
        save_checkpoint(epoch,model, optimizer,is_epoch=True)
        end_epochtime = time.time()
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')

train()





