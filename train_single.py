from __future__ import print_function,division
import os
import os.path

import time
import argparse
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_net2 import singleNet
from layer.voxel_func import *

is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Single-view reconstruct CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/CsgData',
                    help='path to dataset')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--resume', default='./model/csg_single_model.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./model/csg_single_model.pth)')

parser.add_argument('--reset-lr', action="store_true",default=False,
                    help='whether to reset the learning rate')



args=parser.parse_args()

data_rootpath=args.data
resume=args.resume

if is_GPU:
    torch.cuda.set_device(args.gpu)


dataset=singleDataset(data_rootpath)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=single_collate)

model=singleNet()
if is_GPU:
    model.cuda()

critenrion=VoxelLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.5,0.999))
## init lr=0.002


def save_checkpoint(epoch,model,optimizer,is_epoch=False):
    torch.save({
        'model':model.state_dict(),
        'optim':optimizer.state_dict(),
        'epoch':epoch,
    },'./model/csg_single_model.pth')
    if is_epoch:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
        }, './model_epoch/csg_single_model.pth')
    #        print ("save model of epoch{}".format(epoch))


def log(filename,epoch,batch,loss):
    f1=open(filename,'a')
    f1.write('\n in epoch{} batch{} loss={}'.format(epoch,batch,loss))

def lr_decay(optimizer,epoch,step_epoch=20,decay_rate=0.1):
    if epoch % step_epoch == 0:
        for param in optimizer.param_groups:
            param['lr'] *= decay_rate
            f1 = open('log.txt', 'a')
            f1.write('\n in epoch{} learning_rate={}'.format(epoch, param['lr']))
        print ("In epoch:{} learning rate decay".format(epoch))



def train():
    model.train()
    start_epoch = args.start_epoch
    num_epochs = args.epochs

    if os.path.isfile(resume):
        checkoint = torch.load(resume)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])

        if not args.reset_lr:
            optimizer.load_state_dict(checkoint['optim'])
        else:
            print ('reset learning rate')

        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()
        for batch_idx, (imgs, targets) in enumerate(data_loader):
            if is_GPU:
                imgs = Variable(imgs.cuda())
                targets = [Variable(anno.cuda(), requires_grad=False) for anno in targets]
            else:
                imgs = Variable(imgs)
                targets = [Variable(anno, requires_grad=False) for anno in targets]

            t1=time.time()
            outputs = model(imgs)

            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2=time.time()
            print ("in batch:{} loss={} use time:{}s".format(batch_idx, loss.data[0],t2-t1))
            #evaluate()
            if batch_idx%100==0:
                save_checkpoint(epoch, model, optimizer)
                log('csg_SingleTrain_log.txt', epoch, batch_idx, loss.data[0])
        save_checkpoint(epoch,model, optimizer,is_epoch=True)
        end_epochtime = time.time()
        lr_decay(optimizer,epoch)
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')

train()





