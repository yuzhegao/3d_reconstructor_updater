import numpy as np
import os

model_path='/home/gaoyuzhe/PycharmProjects/3Dbuilder/dataset/orgin_binvox'

'''
max_index=595
with open('trainval_8.txt','wb') as fp:
    for index in xrange(max_index):
        os.chdir(model_path)
        binvox_name = '{}.binvox'.format(index)
        if not os.path.exists(binvox_name):
            continue
        for camera_index in xrange(8):
            fp.write('{}_{}\n'.format(index,camera_index))
'''

with open('./dataset/CubeData/test_cube_m.txt','w') as fp:
    for index in xrange(2700):
        ## 2700 of 3000 model to training
        for v1_idx in xrange(8):
            for v2_idx in xrange(13):
                if v1_idx==v2_idx:
                    continue
                fp.write('model_{}_{}_{}\n'.format(index,v1_idx,v2_idx))














