import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from Direction.src.config import *
from Direction.src.dire_data import Data_Gen, generate_case_data, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model import Mesh2FC

tf.enable_eager_execution()

epochs=10000
data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
rf=Rotate_feed(10,data_gen)
    
    
for epoch in range(epochs):
    print('epoch: %d' % (epoch))
    # order = np.arange(case_num)
    # np.random.shuffle(order)
    idx=0
    while(True):
        idx+=1
        print(idx)
        feed_dict = rf.get_feed()
        if feed_dict is not None:
            output = Mesh2FC(feed_dict, CHANNELS, fc_dim=4)
            output = tf.reshape(output, [4])
            loss = pose_estimation_loss(feed_dict['input'], feed_dict['label'], output)
            print(loss)
        else:
            break
            
        
            


