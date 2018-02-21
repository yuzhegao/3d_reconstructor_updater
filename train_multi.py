from __future__ import print_function,division

import os.path
import argparse

import time
from data_prepare.bulid_data import *
from layer.voxel_net2 import MulitUpdateNet
from layer.voxel_func import *

is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Multi-view updater CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/CubeData',
                    help='path to dataset')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--resume', default='./model/latest_model_multi.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./model/latest_model_multi.pth)')
parser.add_argument('--single-model', default='./single_model/t_latest_model.pth', type=str, metavar='PATH',
                    help='path to single model (default: ./single_model/t_latest_model.pth)')
parser.add_argument('--reset-lr', action="store_true",default=False,
                    help='whether to reset the learning rate')



args=parser.parse_args()

data_rootpath=args.data
resume=args.resume

singlemodel_path=args.single_model

if is_GPU:
    torch.cuda.set_device(args.gpu)

dataset=multiDataset(data_rootpath,)
data_loader =torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=multi_collate)

model=MulitUpdateNet(SingleModel=singlemodel_path)
if is_GPU:
    model=model.cuda()

"""
x1=torch.randn(2,1,256,256).cuda()
x2=torch.randn(2,1,256,256).cuda()

x1,x2=Variable(x1),Variable(x2)
out=model(x1,x2,4,7)
print out
"""

critenrion=VoxelL1()
optimizer=torch.optim.Adam([{'params': model.conv1.parameters()},
                            {'params': model.conv2.parameters()},
                            {'params': model.conv3.parameters()},
                            {'params': model.conv4.parameters()},
                            {'params': model.conv5.parameters()},
                            {'params': model.conv4m.parameters()},
                            {'params': model.conv3m.parameters()},
                            {'params': model.conv2m.parameters()},
                            {'params': model.max_pool.parameters()},
                            {'params': model.upsample54.parameters()},
                            {'params': model.upsample43.parameters()},
                            {'params': model.upsample32.parameters()},
                            ],lr=args.lr,betas=(0.5,0.999))

def save_checkpoint(epoch,model,optimizer,is_epoch=False):
    torch.save({
        'model':model.state_dict(),
        'optim':optimizer.state_dict(),
        'epoch':epoch,
    },'./model/latest_model_multi.pth')
    if is_epoch:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
        }, './model_epoch/latest_model_multi.pth')

def log(epoch,batch,loss):
    f1=open('log_MultiTraining.txt','a')
    f1.write('\n in epoch{} batch{} loss={}'.format(epoch,batch,loss))

def lr_decay(optimizer,epoch,step_epoch=20,decay_rate=0.1):
    if epoch % step_epoch == 0 and epoch!=0:
        for param in optimizer.param_groups:
            param['lr'] *= decay_rate
            f1 = open('log_multi.txt', 'a')
            f1.write('\n in epoch{} learning_rate={}'.format(epoch, param['lr']))
        print ("In epoch:{} learning rate decay".format(epoch))



def train():
    model.train()
    start_epoch = 0
    num_epochs = args.epochs
    if os.path.exists(resume):
        if is_GPU:
            checkoint = torch.load(resume)
        else:
            checkoint=torch.load(resume,map_location=lambda storage, loc: storage)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))


    else:
        print("no resume checkpoint to load")
    if args.reset_lr:
        print ('reset learning rate')

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
        model.SingleNet=model.SingleNet.cuda()

    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()
        for batch_idx, (img1s,img2s,target1s,targets,v12s) in enumerate(data_loader):
        ## when training,just go through updater CNN once,so the output ->img2 ,use target2(img2)
            #print "targets shape",targets.size() ##(8L, 64L, 64L, 64L)
            #print 'img1s',img1s.size() ##(8L, 1L, 256L, 256L)
            if is_GPU:
                img1s = Variable(img1s.cuda())
                img2s = Variable(img2s.cuda())
                targets =Variable(targets.cuda())
            else:
                img1s = Variable(img1s)
                img2s = Variable(img2s)
                targets = Variable(targets)

            t1=time.time()

            #print v12s  ##a list of v12 : [(v11,v12),(v21,v22)......]
            outputs = model(img1s,img2s,v12s,target1s)

            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2=time.time()

            print ("in batch:{} loss={} time_cost:{} ".format(batch_idx, loss.data[0],t2-t1))
            #evaluate()
            if batch_idx%100==0 and batch_idx!=0:
                save_checkpoint(epoch, model, optimizer)
                log(epoch, batch_idx, loss.data[0])
        save_checkpoint(epoch,model, optimizer,is_epoch=True)
        end_epochtime = time.time()

        # do not decay learning rate at the begining
        #lr_decay(optimizer,epoch)
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')

train()