import torch
import numpy as np

'''
def singlenet_eval(preds,targets):
    #print 'evaluate the singlenet'
    #print preds.type()
    #print targets.type()

    insect = (targets[preds]).sum()
    union = targets.sum() + preds.sum() - insect
    iou = insect * 1.0 / union

    print 'iou:{} {}/{}'.format(iou,insect,union)
    print 'union = {}+{}'.format(targets.sum(),preds.sum())

'''
def singlenet_eval(pred,target):
    ## pred and target -> torch.Tensor
    print pred.size(),target.size()
    pred,target=pred.numpy().astype(np.float32)>0.5,\
                target.numpy().astype(np.float32)>0.5
    intersect=np.sum(target[pred])
    union=np.sum(pred) + np.sum(target) - intersect
    print 'iou={} pred={}'.format(intersect*1.0/union,np.sum(pred))
    return intersect,intersect*1.0/union
