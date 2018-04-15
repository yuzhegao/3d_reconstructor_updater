from __future__ import print_function,division
import os
import os.path
import shutil
import numpy as np
import torch

import time
import argparse
from data_prepare.author_data import singleDataset,single_collate


#from layer.voxel_net2 import singleNet
#from layer.voxel_deepernet import singleNet_deeper,weights_init
#from layer.voxel_verydeepnet import singleNet_verydeep,weights_init

from layer.unet import single_UNet,weights_init
from torch.autograd import Variable
import torch.nn.functional as F


is_GPU=torch.cuda.is_available()
#torch.set_printoptions(threshold=float('Inf'))

parser = argparse.ArgumentParser(description='Single-view reconstruct CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/chairs/database_64',
                    help='path to dataset')
parser.add_argument('--log', metavar='LOG',default='log.txt',
                    help='filename of log file')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=200000000000000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-step', default=50, type=int, metavar='N',
                    help='number of iter to write log')
parser.add_argument('--test-step', default=1000, type=int, metavar='N',
                    help='number of iter to evaluate ')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')


parser.add_argument('--resume', default='vases_single_unet4.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: csg_single_model.pth)')

args=parser.parse_args()


data_rootpath=args.data
resume='./model/'+args.resume
## args.resume: just the name of checkpoint file
logname=args.log

if is_GPU:
    torch.cuda.set_device(args.gpu)

dataset=singleDataset(data_rootpath)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=single_collate)

model=single_UNet()
if is_GPU:
    model.cuda()

critenrion=torch.nn.NLLLoss()
log_prob=torch.nn.LogSoftmax(dim=1)
prob=torch.nn.Softmax(dim=1)

current_best_IOU=0


def save_checkpoint(epoch,model,optimizer,num_iter):
    global current_best_IOU
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch,
        'best_IOU': current_best_IOU,
        'iter':num_iter,
    }, './model/' + args.resume)




def log(filename,epoch,batch,loss):
    f1=open(filename,'a')
    if epoch == 0 and batch == 100:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    f1.write('\nin epoch{} batch{} loss={} '.format(epoch,batch,loss))


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

    return intersect,intersect*1.0/union

def evaluate(model_test):

    model_test.eval()
    IOUs=0
    total_correct=0

    data_eval = singleDataset(data_rootpath,test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=2, shuffle=True, collate_fn=single_collate)
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

    model_test.train()
    with open(logname,'a') as f:
        f.write('\nthe evaluate average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset))))
    return IOUs*1.0/(len(eval_loader.dataset))



def train():
    global current_best_IOU
    model.train()
    #model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0005)

    start_epoch = args.start_epoch
    num_epochs = args.epochs
    num_iter=0

    if os.path.isfile(resume):
        checkoint = torch.load(resume,map_location={'cuda:0':'cuda:3'})
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        optimizer = checkoint['optim']
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")
    print ('training start!\n')


    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()

        for batch_idx, (imgs, targets) in enumerate(data_loader):
            if num_iter > 1000000:
                exit()

            if num_iter%300000 ==0 and num_iter!=0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.5
                with open(logname, 'a') as f9:
                    f9.write('decay learning rate in iter:{}'.format(num_iter))


            if num_iter%args.log_step==0 and num_iter!=0:
                save_checkpoint(epoch, model, optimizer,num_iter)
                log(logname, epoch, batch_idx, loss.data[0])
            if num_iter%args.test_step==0 and num_iter!=0:
                evaluate(model)

            targets = targets.view(-1, 4096* 64)
            targets=targets.long()

            if is_GPU:
                imgs = Variable(imgs.cuda())
                targets=Variable(targets.cuda())
            else:
                imgs = Variable(imgs)
                targets=Variable(targets)

            t1=time.time()
            outputs = model(imgs)
            outputs = log_prob(outputs)

            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_iter+=1
            t2=time.time()
            print ("in epoch-{} iter-{} loss={} use time:{}s".format(epoch,num_iter, loss.data[0],t2-t1))




        end_epochtime = time.time()
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')
        save_checkpoint(epoch,model,optimizer,num_iter)


train()





