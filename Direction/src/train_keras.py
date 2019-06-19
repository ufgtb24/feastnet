import os
from datetime import datetime

import tensorflow as tf

from Direction.src.config import *
from Direction.src.dire_data import Data_Gen, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model_keras import DirectionModel
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config=config) #1.x
# tf.debugging.set_log_device_placement(True)
# print(tf.executing_eagerly())

keras=tf.keras
optimizer = tf.train.AdamOptimizer() #1.x
# optimizer = keras.optimizers.Adam() #2.x

model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)
# data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz_test')
rot_num=20
rf=Rotate_feed(rot_num,data_gen)
mean_metric = keras.metrics.Mean()


# time_ckpt_dir = None  # where to restore the model
load_time_dir = '/20190619-0839'  # where to restore the model
model_name = '/model-80'
need_save = True


if need_save:
    dir_save = datetime.now().strftime("%Y%m%d-%H%M")
    save_time_dir = '../ckpt/'+dir_save
    # ckpt_dir = '../ckpt/' + dir_save
    os.makedirs(save_time_dir)

var_created=False
for epoch in range(100000):
    # print("epoch ",epoch)
    mean_metric.reset_states()
    epoch_end=False
    while(not epoch_end):
        feed_dict,epoch_end = rf.rotate_case()
        with tf.GradientTape() as tape:
            
            output = model(feed_dict)
            loss = pose_estimation_loss(feed_dict['ori_vertice'], feed_dict['label'], output)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        mean_metric.update_state(loss)
        
        if not var_created:
            # 此时才创建变量
            saver = tf.train.Saver(model.trainable_variables + optimizer.variables())
            if load_time_dir is not None:
                saver.restore(None, '../ckpt' + load_time_dir + model_name)
            var_created = True

    if epoch%40==0:
        print("epoch %d  : %f"%(epoch,mean_metric.result().numpy()))

        if need_save:
            save_path = saver.save(sess=None, save_path=save_time_dir + '/model', global_step=epoch)
            # model.save_weights("../ckpt/ckpt")
            # rut_manager.save(epoch)
            # ckpt.save(os.path.join(ckpt_dir_rut,'model.ckpt'))
