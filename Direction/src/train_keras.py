import os
from datetime import datetime

import tensorflow as tf

from Direction.src.config import *
from Direction.src.dire_data import Data_Gen, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model_keras import DirectionModel
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# tf.enable_eager_execution(config=config) #1.x
# tf.debugging.set_log_device_placement(True)
# print(tf.executing_eagerly())

keras=tf.keras
# optimizer = tf.train.AdamOptimizer() #1.x
optimizer = keras.optimizers.Adam() #2.x

model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)
# data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz_test')
rot_num=20
rf=Rotate_feed(rot_num,data_gen)
mean_metric = keras.metrics.Mean()


# dir_load = None  # where to restore the model
dir_load = '/20190620-1052/rutine'  # where to restore the model
model_name = 'ckpt-920'
need_save = True

# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model,
#                            optimizer_step=tf.train.get_or_create_global_step())
ckpt = tf.train.Checkpoint( optimizer=optimizer, model=model)
if dir_load is not None:
    load_checkpoints_dir = MODEL_PATH + dir_load
    var_file = os.path.join(load_checkpoints_dir, model_name)
    status = ckpt.restore(tf.train.latest_checkpoint(load_checkpoints_dir))
    print(tf.train.list_variables(tf.train.latest_checkpoint(load_checkpoints_dir)))

if need_save:
    dir_save = datetime.now().strftime("%Y%m%d-%H%M")
    ckpt_dir = MODEL_PATH + '/' + dir_save
    os.makedirs(ckpt_dir)
    ckpt_dir_val = ckpt_dir + '/valid'
    ckpt_dir_rut = ckpt_dir + '/rutine'
    os.makedirs(ckpt_dir_val)
    os.makedirs(ckpt_dir_rut)
    val_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_val, max_to_keep=3)
    rut_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_rut, max_to_keep=3)

created=False
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
        # if not loaded:
        # model.load_weights("../ckpt/ckpt")
        # loaded=True
        # print('loss = %f'%(loss))
        mean_metric.update_state(loss)
        status.assert_consumed()
        

    if epoch%40==0:
        print("epoch %d  : %f"%(epoch,mean_metric.result().numpy()))

        if need_save:
            # model.save_weights("../ckpt/ckpt")
            rut_manager.save(epoch)
            # ckpt.save(os.path.join(ckpt_dir_rut,'model.ckpt'))
