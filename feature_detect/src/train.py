from __future__ import division

import os
from datetime import datetime

from feature_detect.src.config import FEAT_CAP, BLOCK_NUM, CHANNELS, ADJ_K, MODEL_PATH
from feature_detect.src.feat_data import Data_Gen
from feature_detect.src.loss_func import loss_func
from common.model import *
import numpy as np




"""
Load dataset 
x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
										  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
										  e.g. [16,10,4] 16 batch, 10 vertice with 4 neib for each
"""

def build_input(block_num):
    adjs=[]
    perms=[]
    input=tf.placeholder(tf.float32,[None,3])
    label=tf.placeholder(tf.float32,[FEAT_CAP,4])
    for i in range(block_num):
        adjs.append(tf.placeholder(tf.int32,[None,ADJ_K]))
        perms.append(tf.placeholder(tf.int32,[None]))
    return {'input':input,'label':label,'adjs':adjs,'perms':perms}


def build_feed_dict(plc,data,iter):
    
    feed_dict={
        plc['input']:data['x'][iter],
        plc['label']:data['y'][iter],
    }
    
    adjs_dict={ adj_plc:data['adj'][iter][idx] for idx,adj_plc in enumerate(plc['adjs'])}
    perms_dict={ perm_plc:data['perm'][iter][idx] for idx,perm_plc in enumerate(plc['perms'])}
    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    return feed_dict
    

plc=build_input(BLOCK_NUM)
output = get_model(plc,CHANNELS, FEAT_CAP )
loss=loss_func(output,plc['label'])
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)


# Train for the dataset

var_list = tf.trainable_variables()
loader = tf.train.Saver(var_list=var_list)
saver_rut = tf.train.Saver( var_list=var_list,max_to_keep=20)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    dir_load=None
    # dir_load = '/20190430-1842/valid'  # where to restore the model
    model_name = 'model.ckpt-6'
    need_save = True
    save_epoch_internal=100
    
    if dir_load is not None:
        load_checkpoints_dir = MODEL_PATH + dir_load
        var_file = os.path.join(load_checkpoints_dir, model_name)
        loader.restore(sess, var_file)  # 从模型中恢复最新变量
    if need_save:
        dir_save = datetime.now().strftime("%Y%m%d-%H%M")
        ckpt_dir = os.path.join(MODEL_PATH ,dir_save)
        os.makedirs(ckpt_dir)
        ckpt_dir_val = os.path.join(ckpt_dir ,'valid')
        ckpt_dir_rut = os.path.join(ckpt_dir ,'rutine')
        os.makedirs(ckpt_dir_val)
        os.makedirs(ckpt_dir_rut)
        writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

    epochs = 10000
    state = [False, '']
    data_fen = Data_Gen('F:/ProjectData/mesh_feature/tooth/save_npz/back')
    data, case_num = data_fen.load_pkg(state)
    
    sum_train = {'loss': 0}
    for epoch in range(epochs):
        print('epoch: %d'%(epoch))

        order=np.arange(case_num)
        np.random.shuffle(order)
        for e,i in enumerate(order):
    
            feed_dict=build_feed_dict(plc,data,i)

            loss_out,_=sess.run([loss,train_step],feed_dict=feed_dict)
            sum_train['loss'] = (sum_train['loss'] * e + loss_out) / (e + 1)

            # print(loss_out)
        if need_save:
            if (epoch + 1) % save_epoch_internal == 0:
                saver_rut.save(sess, ckpt_dir_rut + "/model.ckpt", global_step=epoch + 1)

        print('epoch train loss = : %.5f'%(sum_train['loss'] ))
        
    
    
    


