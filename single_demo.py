import os
import os.path

import time
from data_prepare.bulid_data import singleDataset,single_collate
from layer.voxel_net2 import singleNet
from layer.voxel_func import *
from utils.utils_rw import *


def IOU(model1,model2):
    insect= np.sum(model1[model2])
    union= np.sum(model1)+np.sum(model2)-insect
    return insect*1.0/union

def eval_model(model_name):
    with open('./dataset/CsgData/trainval/binvox/'+model_name+'.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_3d_array(f)

    with open('./demo/'+model_name+'.binvox', 'rb') as f1:
        # m1 = read_as_coord_array(f)
        m2 = read_as_3d_array(f1)

    # print m1.data,m2.data
    print IOU(m1.data, m2.data)



np.set_printoptions(threshold='nan')
resume='./model_epoch/csg_single_model.pth'

model=singleNet()

is_GPU=torch.cuda.is_available()
if is_GPU:
    torch.cuda.set_device(0)

if os.path.exists(resume):
    if is_GPU:
        model.load_state_dict(torch.load(resume)['model'])
    else:
        model.load_state_dict(torch.load(resume,map_location=lambda storage, loc: storage)['model'])

    print 'load trained model success'

data_rootpath='./dataset/CsgData'
dataset=singleDataset(data_rootpath)

for i in xrange(210,213):
    img_id, test_img = dataset.pull_img(i)
    # test_img = np.array(test_img).astype(np.float32)
    # cv2.imshow('mat',test_img)
    # cv2.waitKey(1000)
    print img_id

    test_img = test_img[np.newaxis, :]
    test_img = test_img[np.newaxis, :]

    #print test_img.shape
    img = torch.from_numpy(test_img).type(torch.FloatTensor)
    img = Variable(img)
    init = time.time()
    up = model(img)
    end = time.time()

    result = up.data[0, :, :, :].numpy()
    print result
    print "each forward use:{}s".format(end - init)
    result = result >= 0.5

    with open('0_0.binvox', 'rb') as f:
        # m1 = read_as_coord_array(f)
        m1 = read_as_orginal_coord(f)

    m1.data = result.transpose(2,1,0)
    with open('./demo/'+img_id+'.binvox', 'wb')as f1:
        write(m1, f1)
    eval_model(img_id)



