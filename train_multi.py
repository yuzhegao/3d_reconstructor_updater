from __future__ import print_function,division

import os.path
import argparse

import time
from data_prepare.bulid_data import *
from layer.voxel_verydeepnet import MulitUpdateNet_verydeep,weights_init
from layer.voxel_func import CrossEntropy_loss
from torch.autograd import Variable


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Multi-view updater CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/CsgData',
                    help='path to dataset')

parser.add_argument('--data-name', default='csg', type=str, metavar='PATH',
                    help='name of dataset (default: csg)')
## this arg: name of log, data list file(.txt)

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
parser.add_argument('--gamma', default=0.7, type=float,
                    metavar='GM,', help='param of cross entropy loss')

parser.add_argument('--resume', default='latest_model_multi.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: latest_model_multi.pth)')
parser.add_argument('--single-model', default='./single_model/csg_single_train_ce_server_81.pth', type=str, metavar='PATH',
                    help='path to single model (default: ./single_model/t_latest_model.pth)')



args=parser.parse_args()

data_rootpath=args.data
resume='./model/'+args.resume
## args.resume: just the name of checkpoint file

logfile=args.data_name+'_multi_train.txt'
print ('logfile name:{}'.format(logfile))

singlemodel_path=args.single_model

if is_GPU:
    torch.cuda.set_device(args.gpu)

dataset=multiDataset(data_rootpath,data_name=args.data_name)
data_loader =torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=multi_collate)

model=MulitUpdateNet_verydeep()
if is_GPU:
    model=model.cuda()

"""
x1=torch.randn(2,1,256,256).cuda()
x2=torch.randn(2,1,256,256).cuda()

x1,x2=Variable(x1),Variable(x2)
out=model(x1,x2,4,7)
print out
"""

#critenrion=VoxelL1()
critenrion=CrossEntropy_loss()
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
    },'./model/'+args.resume)
    if is_epoch:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
        }, './model_epoch/'+args.resume)

def log(epoch,batch,loss):
    f1=open(logfile,'a')
    if epoch==0 and batch==100:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    f1.write('\nin epoch{} batch{} loss={}'.format(epoch,batch,loss))

def eval_iou(pred,target):
    #print pred.size(),target.size()
    pred,target=pred.cpu().numpy().astype(np.float32)>0.5,\
                target.cpu().numpy().astype(np.float32)>0.5
    intersect=np.sum(target[pred])
    union=np.sum(pred) + np.sum(target) - intersect
    print ("output occupy sum:{}".format(np.sum(pred)))
    print ('iou:{}'.format(intersect*1.0/union))

    return intersect,intersect*1.0/union


def train():
    model.train()
    model.apply(weights_init)

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
    else:
        print ('Warning: single model do not exist!')
        exit()




    if is_GPU:
        model.SingleNet=model.SingleNet.cuda()

    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()
        for batch_idx, (img1s,img2s,target1s,targets,v12s) in enumerate(data_loader):
        ## when training,just go through updater CNN once,so the output ->img2 ,use target2(img2)

            if is_GPU:
                img1s = Variable(img1s.cuda())
                img2s = Variable(img2s.cuda())
                targets =Variable(targets.cuda())
            else:
                img1s = Variable(img1s)
                img2s = Variable(img2s)
                targets = Variable(targets)

            t1=time.time()

            """
            if True:
                pred_v1=model.SingleNet(img1s)
                eval_iou(pred_v1.data,target1s) ##just test singleNet
            """

            #print v12s  ##a list of v12 : [(v11,v12),(v21,v22)......]
            outputs = model(img1s,img2s,v12s,target1s)

            loss = critenrion(outputs, targets,gamma=args.gamma)

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