import os
import os.path

import time
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_net2 import singleNet
from layer.voxel_func import *
from utils.utils_rw import *

resume='./model_epoch/csg_single_model.pth'

def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union

def eval_iou(pred,target):
    #print pred.size(),target.size()
    pred,target=pred.numpy().astype(np.float32)>0.5,\
                target.numpy().astype(np.float32)>0.5
    intersect=np.sum(target[pred])
    union=np.sum(pred) + np.sum(target) - intersect

    return intersect,intersect*1.0/union


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
data_rootpath='./dataset/CsgData'

model=singleNet()
if is_GPU:
    model.cuda()

def evaluate():
    ## calculate the average IOU between pred and gt
    if os.path.isfile(resume):
        if is_GPU:
            checkoint = torch.load(resume)
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
                    batch_size=1, shuffle=True, collate_fn=single_collate)

    for batch_idx,(imgs, targets) in enumerate(eval_loader):
        if is_GPU:
            imgs = Variable(imgs.cuda())
            targets = [Variable(anno.cuda(),requires_grad=False) for anno in targets]
        else:
            imgs = Variable(imgs)
            targets = [Variable(anno, requires_grad=False) for anno in targets]
        t1=time.time()
        outputs=model(imgs)

        #occupy=(outputs.data>0.5)if is_GPU else (outputs.data>0.5).type(torch.FloatTensor)
        #print occupy.sum()
        #print 'c1',torch.sum(outputs.data)
        occupy = (outputs.data > 0.5)  ## ByteTensor
        #print 'c1', torch.sum(occupy)


        for idx,target in enumerate(targets):
            #correct+=(occupy[idx].eq(target.data)).sum()
            #print target.data.type() ##(64L, 64L, 64L) FloatTensor
            #print occupy[idx].type() ##(64L, 64L, 64L) ByteTensor

            '''
            insect=(target.data[occupy[idx]]).sum()
            union=target.data.sum()+occupy[idx].sum()-insect
            iou=insect*1.0/union
            IOUs+=iou
            '''
            insect,iou=eval_iou(occupy[idx],target.data)
            IOUs += iou

            total_correct += insect

        t2=time.time()
        print 'in batch{} cost{}s'.format(batch_idx,t2-t1)

    #print 'correct num:{}'.format(total_correct)
    print 'the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset)))
    print 'the average iou:{}'.format(IOUs*1.0/(len(eval_loader.dataset)))

evaluate()