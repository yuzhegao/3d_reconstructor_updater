import numpy as np
import os

#max_index=17000
max_index=170

with open('test_csg_s.txt','w') as fp:
    for index in xrange(max_index):
        binvox_name = '{}.binvox'.format(index)
        for camera_index in xrange(8):
            fp.write('{}_{}\n'.format(index,camera_index))

















