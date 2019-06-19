import os

from Direction.src.config import *
from Direction.src.dire_data import process_data
from common.model_keras import DirectionModel
import tensorflow as tf

# [B,N_INPUT,C]
from common.place_holder_ops import *
from common.write_graph import write_pb

plc,input_names,input_types=build_plc_b(BLOCK_NUM,adj_dim=ADJ_K)

# optimizer = tf.train.AdamOptimizer() #1.x
model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

output = model(plc)
output=tf.identity(output,'output_node')
init = tf.global_variables_initializer()

saver = tf.train.Saver(model.trainable_variables)

data_path = "F:/ProjectData/mesh_direction/2aitest/low"

load_time_dir = '20190619-0846'  # where to restore the model
ckpt_file = 'model-3240'
need_freeze=True

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, os.path.join(CKPT_PATH,load_time_dir,ckpt_file))
    if need_freeze:
        input_name=[key for key in plc.keys()]
        write_pb(sess,os.path.join(CKPT_PATH,load_time_dir),ckpt_file,input_names,input_types)
    else:
        # status.initialize_or_restore(sess)
        X, Adjs, Perms=process_data(data_path, 'case_list.txt')
        for x,adjs,perms in zip(X, Adjs, Perms):
            feed_dict=build_feed_dict_b(plc,x,adjs,perms)
            result=sess.run(output,feed_dict=feed_dict)
            print(result.shape)
    

