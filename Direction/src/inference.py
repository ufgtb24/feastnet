import os

from Direction.src.config import *
from Direction.src.dire_data import process_data
from common.model_keras import DirectionModel
import tensorflow as tf

# [B,N_INPUT,C]
from common.place_holder_ops import *

plc=build_plc_b(BLOCK_NUM,adj_dim=ADJ_K)
optimizer = tf.train.AdamOptimizer() #1.x
model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

dir_load = '/20190617-1113/rutine'  # where to restore the model
model_name = 'model.ckpt-1'
load_checkpoints_dir = MODEL_PATH + dir_load
var_file = os.path.join(load_checkpoints_dir, model_name)
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)

root.restore(tf.train.latest_checkpoint(var_file))
output = model(plc)
data_path = "F:/ProjectData/mesh_direction/2aitest/low"

with tf.Session() as sess:
    X, Adjs, Perms=process_data(data_path, 'case_list.txt')
    for x,adjs,perms in zip(X, Adjs, Perms):
        feed_dict=build_feed_dict_b(plc,x,adjs,perms)
        result=sess.run(output,feed_dict=feed_dict)


