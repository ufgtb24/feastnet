import os
from datetime import datetime

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion
import numpy as np

from Direction.src.config import *
from Direction.src.data_gen import Data_Gen
from Direction.src.loss import pose_estimation_loss
from common.model import Mesh2FC
from common.place_holder_ops import build_plc, build_feed_dict

plc=build_plc(BLOCK_NUM,label_shape=[4],adj_dim=ADJ_K)
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
    # dir_load=None
    dir_load = '/20190524-1818/rutine'  # where to restore the model
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
    state = [False, '']
    data_fen = Data_Gen('F:/ProjectData/mesh_feature/tooth/save_npz/back')
    data, case_num = data_fen.load_pkg(state)
    
    
    def rotate(data):
        data['x']
    
    
    
    
    sum_train = {'loss': 0}
    for epoch in range(epochs):
        print('epoch: %d' % (epoch))
        
        order = np.arange(case_num)
        np.random.shuffle(order)
        for e, i in enumerate(order):
            feed_dict = build_feed_dict(plc, data, i)
            
            loss_out, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            sum_train['loss'] = (sum_train['loss'] * e + loss_out) / (e + 1)
            
            # print(loss_out)
        if need_save:
            if (epoch + 1) % save_epoch_internal == 0:
                saver_rut.save(sess, ckpt_dir_rut + "/model.ckpt", global_step=epoch + 1)
        
        print('epoch train loss = : %.5f' % (sum_train['loss']))



