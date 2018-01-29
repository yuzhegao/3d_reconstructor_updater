import os
import os.path

import time
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_net2 import singleNet
from layer.voxel_func import *

is_GPU=torch.cuda.is_available()

if is_GPU:
    torch.cuda.set_device(3)

data_rootpath='./dataset/CubeData'
resume='./model/latest_model.pth'

dataset=singleDataset(data_rootpath)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=single_collate)

model=singleNet()
if is_GPU:
    model.cuda()

critenrion=VoxelLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.002,betas=(0.5,0.999))
## init lr=0.002


def save_checkpoint(epoch,model,optimizer,is_epoch=False):
    torch.save({
        'model':model.state_dict(),
        'optim':optimizer.state_dict(),
        'epoch':epoch,
    },'./model/latest_model.pth')
    if is_epoch:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
        }, './model_epoch/latest_model.pth')
    #        print ("save model of epoch{}".format(epoch))

def evaluate():
    if os.path.isfile(resume):
        checkoint = torch.load(resume)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        optimizer.load_state_dict(checkoint['optim'])
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("no resume checkpoint to load")

    model.eval()
    correct=0
    data_eval = singleDataset(data_rootpath, test=True)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                                              batch_size=4, shuffle=True, collate_fn=single_collate)

    for batch_idx,(imgs, targets) in enumerate(eval_loader):
        if is_GPU:
            imgs = Variable(imgs.cuda())
            targets = [Variable(anno.cuda(),requires_grad=False) for anno in targets]
        else:
            imgs = Variable(imgs)
            targets = [Variable(anno, requires_grad=False) for anno in targets]
        outputs=model(imgs)

        occupy=(outputs.data>0.5).type(torch.cuda.FloatTensor)if is_GPU else (outputs.data>0.5).type(torch.FloatTensor)
        print occupy.sum()
        for idx,target in enumerate(targets):
            correct+=(occupy[idx].eq(target.data)).sum()
    print 'correct num:{}'.format(correct)
    print 'the average correct rate:{}'.format(correct*1.0/(len(eval_loader.dataset)))

def log(epoch,batch,loss):
    f1=open('log.txt','a')
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
    start_epoch = 0
    num_epochs = 200
    if os.path.isfile(resume):
        checkoint = torch.load(resume)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        optimizer.load_state_dict(checkoint['optim'])
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
            print "in batch:{} loss={}".format(batch_idx, loss.data[0],t2-t1)
            #evaluate()
            if batch_idx%100==0:
                save_checkpoint(epoch, model, optimizer)
                log('log.txt', epoch, batch_idx, loss.data[0])
        save_checkpoint(epoch,model, optimizer,is_epoch=True     )
        end_epochtime = time.time()
        lr_decay(optimizer,epoch)
        print '--------------------------------------------------------'
        print 'in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime)
        print '--------------------------------------------------------'

train()
#evaluate()





