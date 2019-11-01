import numpy as np
import os
from datetime import datetime

import tensorflow as tf

from feature_detect.src.config import *
from feature_detect.src.feat_data_keras import Data_Gen, Rotate_feed
from common.extract_model import ExtractModel
from feature_detect.src.loss_func import loss_func

keras=tf.keras

# ###  enable GPU growth !!
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

optimizer = keras.optimizers.Adam() #2.x

model=ExtractModel(CHANNELS, coarse_level=C_LEVEL, fc_dim=FEAT_CAP*3)
data_gen = Data_Gen('F:/ProjectData/mesh_feature/Case_npz/back')

rf = Rotate_feed(
    rot_num=2,
    rot_range=[np.pi / 18., np.pi / 18., np.pi / 18.],
    angle_fixed=True,
    data_gen=data_gen
)
mean_metric = keras.metrics.Mean()


# dir_load = None  # where to restore the model
dir_load = '/20191031-1620/rutine'  # where to restore the model
model_name = 'ckpt-17'
need_save = True

# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model,
#                            optimizer_step=tf.train.get_or_create_global_step())
ckpt = tf.train.Checkpoint( optimizer=optimizer, model=model)
if dir_load is not None:
    load_checkpoints_dir = CKPT_PATH + dir_load
    var_file = os.path.join(load_checkpoints_dir, model_name)
    status = ckpt.restore(tf.train.latest_checkpoint(load_checkpoints_dir))
    print(tf.train.list_variables(tf.train.latest_checkpoint(load_checkpoints_dir)))

if need_save:
    dir_save = datetime.now().strftime("%Y%m%d-%H%M")
    ckpt_dir = CKPT_PATH + '/' + dir_save
    os.makedirs(ckpt_dir)
    ckpt_dir_val = ckpt_dir + '/valid'
    ckpt_dir_rut = ckpt_dir + '/rutine'
    os.makedirs(ckpt_dir_val)
    os.makedirs(ckpt_dir_rut)
    val_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_val, max_to_keep=3)
    rut_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_rut, max_to_keep=3)
    writer = tf.summary.create_file_writer(ckpt_dir_rut)

keras.datasets.mnist.load_data()


for epoch in range(100000):
    # print("epoch ",epoch)
    mean_metric.reset_states()
    epoch_end=False
    while(not epoch_end):
        feed_dict,epoch_end = rf.rotate_case()
        with tf.GradientTape() as tape:
            
            output = model(feed_dict)
            output = tf.reshape(output, [-1,FEAT_CAP, 3])
            loss = loss_func(output,feed_dict['label'],feed_dict['mask'])


        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        mean_metric.update_state(loss)
        if dir_load is not None:
            status.assert_consumed()
        

    if epoch%1==0:
        epoch_loss=mean_metric.result().numpy()
        print("epoch %d  : %f"%(epoch,epoch_loss))

        if need_save:
            with writer.as_default():
                tf.summary.scalar("loss", epoch_loss, step=epoch)
                writer.flush()
            # model.save_weights("../ckpt/ckpt")
            rut_manager.save(epoch)
            # ckpt.save(os.path.join(ckpt_dir_rut,'model.ckpt'))

