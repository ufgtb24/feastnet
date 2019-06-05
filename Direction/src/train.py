import os
from datetime import datetime

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion
import numpy as np

from Direction.src.config import *
from Direction.src.dire_data import Data_Gen, generate_case_data, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model import Mesh2FC, Rotate
from common.place_holder_ops import build_plc, build_feed_dict

plc=build_plc(BLOCK_NUM,label_shape=[4],adj_dim=ADJ_K)
ori_data=tf.placeholder(tf.float32, [None, 3])
expand_x,expand_y= Rotate(ori_data,rot_num)

output = Mesh2FC(plc, CHANNELS, fc_dim=4)
output = tf.reshape(output, [4])
loss = pose_estimation_loss(plc['input'],plc['label'],output)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

var_list = tf.trainable_variables()
loader = tf.train.Saver(var_list=var_list)
saver_rut = tf.train.Saver(var_list=var_list, max_to_keep=20)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    dir_load=None
    # dir_load = '/20190524-1818/rutine'  # where to restore the model
    model_name = 'model.ckpt-4300'
    need_save = True
    save_epoch_internal = 500
    
    if dir_load is not None:
        load_checkpoints_dir = MODEL_PATH + dir_load
        var_file = os.path.join(load_checkpoints_dir, model_name)
        loader.restore(sess, var_file)  # 从模型中恢复最新变量
    if need_save:
        dir_save = datetime.now().strftime("%Y%m%d-%H%M")
        ckpt_dir = os.path.join(MODEL_PATH, dir_save)
        os.makedirs(ckpt_dir)
        ckpt_dir_val = os.path.join(ckpt_dir, 'valid')
        ckpt_dir_rut = os.path.join(ckpt_dir, 'rutine')
        os.makedirs(ckpt_dir_val)
        os.makedirs(ckpt_dir_rut)
        writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
    
    epochs = 10000
    data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')
    rf = Rotate_feed(10, data_gen)
    
    sum_train = {'loss': 0}
    for epoch in range(epochs):
        print('epoch: %d' % (epoch))
        epoch_end=False
        idx=0
        while(not epoch_end):
            data, epoch_end = rf.get_data()
            rot_x, rot_y = sess.run([expand_x,expand_y], feed_dict={
                ori_data:data['x']
            })
    
            rot_data=data
            for i in range(rot_num):
                rot_data['x']=rot_x[i]
                rot_data['y']=rot_y[i]
                feed_dict = build_feed_dict(plc, rot_data, i)
                loss_out, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                sum_train['loss'] = (sum_train['loss'] * idx + loss_out) / (idx + 1)
            idx+=1
                # print(loss_out)
        if need_save:
            if (epoch + 1) % save_epoch_internal == 0:
                saver_rut.save(sess, ckpt_dir_rut + "/model.ckpt", global_step=epoch + 1)
        
        print('epoch train loss = : %.5f' % (sum_train['loss']))
    
    
