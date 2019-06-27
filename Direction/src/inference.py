import os


from Direction.src.config import *
from Direction.src.dire_data import process_data
from Direction.src.freeze_wrapper import write_pb
from Direction.src.plc import *
from common.model_keras import DirectionModel

tf.compat.v1.disable_eager_execution()

plc,input_names=build_plc(BLOCK_NUM,adj_dim=ADJ_K)

# optimizer = tf.train.AdamOptimizer() #1.x
model=DirectionModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

load_time_dir = '20190620-1412/rutine'  # where to restore the model
ckpt_file = 'ckpt-1440'


output = model(plc)
output=tf.identity(output,'output_node')

ckpt_full_dir = os.path.join(CKPT_PATH, load_time_dir)
ckpt_full_path = os.path.join(ckpt_full_dir, ckpt_file)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(ckpt_full_path)
# status.assert_consumed()  optimizer is not built, so this assert won't pass

status.assert_existing_objects_matched()
# init = tf.global_variables_initializer()

# saver=tf.compat.v1.train.Saver(model.trainable_variables)

data_path = "F:/ProjectData/mesh_direction/2aitest/low"

need_freeze=False

with tf.compat.v1.Session() as sess:
    status.initialize_or_restore(sess)
    if need_freeze:
        import shutil
        shutil.rmtree('../freeze_output')
        tf.compat.v1.saved_model.simple_save(sess, '../freeze_output', input_names, {'output_node': output})
        write_pb('../freeze_output')
    else:
        X, Adjs, Perms=process_data(data_path, 'case_test.txt')
        for x,adjs,perms in zip(X, Adjs, Perms):
            feed_dict=build_feed_dict(plc,x,adjs,perms)
            result=sess.run(output,feed_dict=feed_dict)
            print(result.shape)
    

