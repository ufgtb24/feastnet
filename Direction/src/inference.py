import os

from Direction.src.config import *
from Direction.src.dire_data import process_data
from common.model_keras import DirectionModel
import tensorflow as tf

# [B,N_INPUT,C]
from common.place_holder_ops import *
# tf.compat.v1.disable_eager_execution()

plc=build_plc_b(BLOCK_NUM,adj_dim=ADJ_K)
# optimizer = tf.train.AdamOptimizer() #1.x
model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

dir_load = '/20190620-1056/rutine'  # where to restore the model
model_name = 'ckpt-120'
load_checkpoints_dir = MODEL_PATH + dir_load
var_file = os.path.join(load_checkpoints_dir, model_name)
checkpoint = tf.train.Checkpoint(model=model)

status = checkpoint.restore(var_file)
# status.assert_consumed()

output = model(plc)
init = tf.global_variables_initializer()

# saver = tf.train.Saver(model.trainable_variables)

data_path = "F:/ProjectData/mesh_direction/2aitest/low"

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "../ckpt/model.ckpt")

    # status.initialize_or_restore(sess)
    X, Adjs, Perms=process_data(data_path, 'case_list.txt')
    for x,adjs,perms in zip(X, Adjs, Perms):
        feed_dict=build_feed_dict_b(plc,x,adjs,perms)
        result=sess.run(output,feed_dict=feed_dict)
        print(result.shape)


