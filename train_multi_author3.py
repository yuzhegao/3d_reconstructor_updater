from __future__ import print_function,division
import os
import os.path
import shutil
import time
import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

""" dataset """
from data_prepare.author_data import multiDataset,multi_collate

""" network """
from layer.unet import multi_UNet,single_UNet

is_GPU=torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Single-view reconstruct CNN Training')
parser.add_argument('--data', metavar='DIR',default='./dataset/chairs/database_64',
                    help='path to dataset')
parser.add_argument('--num_model', metavar='N',type=int,default=400,
                    help='num of model in dataset')
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
parser.add_argument('--single-model', default='./single_model/unet4_chair.pth', type=str, metavar='PATH',
                    help='path to single model ')

parser.add_argument('--resume', default='unet4_vase.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint ')

args=parser.parse_args()


data_rootpath=args.data
resume='./model/'+args.resume
logname=args.log

if is_GPU:
    torch.cuda.set_device(args.gpu)

dataset=multiDataset(data_rootpath,num_model=args.num_model)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=multi_collate)

model=multi_UNet()
if is_GPU:
    model.cuda()


optimizer = torch.optim.Adam([{'params': model.conv1.parameters()},
                            {'params': model.conv2.parameters()},
                            {'params': model.conv3.parameters()},
                            {'params': model.conv4.parameters()},
                            {'params': model.conv5.parameters()},
                            {'params': model.conv6.parameters()},
                            {'params': model.conv7.parameters()},
                            {'params': model.conv8.parameters()},

                            {'params': model.upsample87.parameters()},
                            {'params': model.upsample76.parameters()},
                            {'params': model.upsample65.parameters()},
                            {'params': model.upsample54.parameters()},
                            {'params': model.upsample43.parameters()},
                            {'params': model.upsample32.parameters()},
                            ], lr=args.lr, betas=(0.5, 0.999))


critenrion=torch.nn.NLLLoss()
log_prob=torch.nn.LogSoftmax(dim=1)
prob=torch.nn.Softmax(dim=1)

current_best_IOU=0


def save_checkpoint(epoch,model,num_iter):
    global current_best_IOU
    torch.save({
        'model': model.state_dict(),
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

    data_eval = multiDataset(data_rootpath,args.num_model,test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=2, shuffle=True, collate_fn=multi_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx, (img1s, img2s, target1s, targets, v12s) in enumerate(eval_loader):

        if is_GPU:
            img1s = Variable(img1s.cuda())
            img2s = Variable(img2s.cuda())
            targets = Variable(targets.cuda())
        else:
            img1s = Variable(img1s)
            img2s = Variable(img2s)
            targets = Variable(targets)


        t1 = time.time()
        print (img1s.data.size())
        outputs = model(img1s, img2s, v12s)
        outputs=prob(outputs)

        _,occupy = torch.max(outputs.data,dim=1)
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

    start_epoch = args.start_epoch
    num_epochs = args.epochs
    num_iter=0

    if os.path.isfile(resume):
        checkoint = torch.load(resume,map_location={'cuda:0':'cuda:1'})
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    if os.path.exists(args.single_model):
        t1 = time.time()
        if is_GPU:
            checkoint = torch.load(args.single_model,map_location={'cuda:0':'cuda:1'})
        else:
            checkoint = torch.load(args.single_model, map_location=lambda storage, loc: storage)

        model.single_net.load = model.single_net.load_state_dict(checkoint['model'])
        t2 = time.time()
        print ('singleNetwork load resume model from epoch{} use {}s'.format(checkoint['epoch'], t2 - t1))
    else:
        print ('Warning: single model do not exist!')
        exit()


    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()

        for batch_idx, (img1s,img2s,target1s,targets,v12s)  in enumerate(data_loader):
            if num_iter > 1000000:
                exit()

            if num_iter%args.test_step==0 and num_iter!=0:
                evaluate(model)

            targets = targets.view(-1, 4096* 64)
            targets=targets.long()

            if is_GPU:
                img1s = Variable(img1s.cuda())
                img2s = Variable(img2s.cuda())
                targets=Variable(targets.cuda())
            else:
                img1s = Variable(img1s)
                img2s = Variable(img2s)
                targets=Variable(targets)
            """
            if True:
                pred_v1=model.single_net(img1s)
                outputs = prob(pred_v1)
                _, occupy = torch.max(outputs.data, dim=1)
                target1s = target1s.view(-1, 4096 * 64)

                print (occupy.size(),target1s.size())
                intersect,iou=eval_iou(occupy,target1s) ##just test single_net
                print (intersect,iou)
            exit()
            """


            t1=time.time()
            outputs = model(img1s,img2s,v12s)
            outputs = log_prob(outputs)

            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_iter+=1
            t2=time.time()
            if num_iter%(args.log_step*10)==0 and num_iter!=0:
                save_checkpoint(epoch, model,num_iter)
            if num_iter%(args.log_step)==0 and num_iter!=0:
                log(logname, epoch, num_iter, loss.data[0])
            print ("in epoch-{} iter-{} loss={} use time:{}s".format(epoch,num_iter, loss.data[0],t2-t1))

        end_epochtime = time.time()
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')
        save_checkpoint(epoch,model,num_iter)

train()
