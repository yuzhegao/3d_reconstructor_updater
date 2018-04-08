from __future__ import print_function,division
import os
import os.path
import shutil
import numpy as np
import torch

import time
import argparse
#from data_prepare.bulid_data import singleDataset,single_collate
from data_prepare.build_data_auther import singleDataset,single_collate


#from layer.voxel_net2 import singleNet
#from layer.voxel_deepernet import singleNet_deeper,weights_init
#from layer.voxel_verydeepnet import singleNet_verydeep,weights_init

from layer.unet import single_UNet,weights_init
#from layer.voxel_func import CrossEntropy_loss
from torch.autograd import Variable
import torch.nn.functional as F


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

parser.add_argument('--data-name', default='authorvase', type=str, metavar='PATH',
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

#model=single_UNet()
model=single_UNet()
if is_GPU:
    model.cuda()

#critenrion=softmax_loss()
critenrion=torch.nn.NLLLoss()
log_prob=torch.nn.LogSoftmax(dim=1)

optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.5,0.999))
current_best_IOU=0


def save_checkpoint(epoch,model,optimizer):
    global current_best_IOU
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch,
        'best_IOU': current_best_IOU,
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

    data_eval = singleDataset(data_rootpath,data_name=args.data_name,test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=2, shuffle=True, collate_fn=single_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx,(imgs, targets) in enumerate(eval_loader):
        targets=targets.view(-1,1,4096,64)
        if is_GPU:
            imgs = Variable(imgs.cuda())
            targets = [Variable(anno.cuda(),requires_grad=False) for anno in targets]
        else:
            imgs = Variable(imgs)
            targets = [Variable(anno, requires_grad=False) for anno in targets]
        outputs=model_test(imgs)
        #outputs=F.softmax(outputs,dim=1)

        #occupy = (outputs.data[:,1] > 0.5)  ## ByteTensor
        occupy = (outputs.data > 0.5)


        for idx,target in enumerate(targets):

            insect,iou=eval_iou(occupy[idx],target.data)
            IOUs += iou

            total_correct += insect


    #print 'correct num:{}'.format(total_correct)
    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))
    print ('the average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset))))

    model_test.train()
    with open(logfile,'a') as f:
        f.write('\nthe evaluate average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset))))
    return IOUs*1.0/(len(eval_loader.dataset))



def train():
    global current_best_IOU
    model.train()
    #model.apply(weights_init)

    start_epoch = args.start_epoch
    num_epochs = args.epochs

    if os.path.isfile(resume):
        checkoint = torch.load(resume,map_location={'cuda:0':'cuda:3'})
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
       # current_best_IOU=checkoint['best_IOU']

        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    print ('training start!\n')
    for epoch in xrange(start_epoch,num_epochs):
        init_epochtime = time.time()


        for batch_idx, (imgs, targets) in enumerate(data_loader):
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
            #print (outputs.data.size())

            #loss = critenrion(outputs, targets,gamma=0.5)
            loss = critenrion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2=time.time()
            print ("in batch:{} loss={} use time:{}s".format(batch_idx, loss.data[0],t2-t1))

            if batch_idx%args.log_step==0 and batch_idx!=0:
                save_checkpoint(epoch, model, optimizer)
                log(logfile, epoch, batch_idx, loss.data[0])
                """
                current_iou=evaluate(model)
                if current_iou>current_best_IOU:
                    current_best_IOU=current_iou
                    if os.path.exists('./model_epoch/'+args.resume):
                        os.remove('./model_epoch/'+args.resume)
                    shutil.copy(resume,'./model_epoch/'+args.resume)
                    """

        end_epochtime = time.time()
        print ('--------------------------------------------------------')
        print ('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print ('--------------------------------------------------------')
        save_checkpoint(epoch,model,optimizer)
        """
        if epoch%1==0:
            current_iou=evaluate(model)
            if current_iou>current_best_IOU:
                current_best_IOU=current_iou
                if os.path.exists('./model_epoch/'+args.resume):
                    os.remove('./model_epoch/'+args.resume)
                shutil.copy(resume,'./model_epoch/'+args.resume)"""


train()





