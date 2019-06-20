import os

from tensorflow.python.tools.freeze_graph import freeze_graph

from Direction.src.config import *
from Direction.src.dire_data import process_data
from common.model_keras import DirectionModel
import tensorflow as tf

# [B,N_INPUT,C]
from common.place_holder_ops import *
from common.write_graph import write_pb
tf.compat.v1.disable_eager_execution()

plc,input_names,input_types=build_plc_b(BLOCK_NUM,adj_dim=ADJ_K)

# optimizer = tf.train.AdamOptimizer() #1.x
model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

load_time_dir = '20190620-1412/rutine'  # where to restore the model
ckpt_file = 'ckpt-1440'

# dir_load = '/20190620-1412/rutine'  # where to restore the model
# model_name = 'ckpt-1440'

ckpt_full_dir = os.path.join(CKPT_PATH, load_time_dir)
ckpt_full_path = os.path.join(ckpt_full_dir, ckpt_file)
checkpoint = tf.train.Checkpoint(model=model)

status = checkpoint.restore(ckpt_full_path)
status.assert_existing_objects_matched()
# status.assert_consumed()

output = model(plc)
output=tf.identity(output,'output_node')
# init = tf.global_variables_initializer()


data_path = "F:/ProjectData/mesh_direction/2aitest/low"

need_freeze=False

with tf.compat.v1.Session() as sess:
    status.initialize_or_restore(sess)
    print(tf.train.list_variables(tf.train.latest_checkpoint(ckpt_full_dir)))

    # sess.run(init)
    if need_freeze:
        input_name=[key for key in plc.keys()]
        write_pb(sess,os.path.join(CKPT_PATH,load_time_dir),ckpt_file,input_names,input_types)
    else:
        # status.initialize_or_restore(sess)
        X, Adjs, Perms=process_data(data_path, 'case_test.txt')
        for x,adjs,perms in zip(X, Adjs, Perms):
            feed_dict=build_feed_dict_b(plc,x,adjs,perms)
            result=sess.run(output,feed_dict=feed_dict)
            print(result.shape)
    

