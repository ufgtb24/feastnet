# %%

import os

import numpy as np
import tensorflow as tf
import trimesh
import tensorboard
from common.extract_model import ExtractModel
from common.freeze_wrapper import write_pb
from feature_detect.src.config import *
from feature_detect.src.feat_data_keras import Data_Gen, Rotate_feed

from tensorboard.plugins.mesh import summary as mesh_summary
from common.plc import build_plc, build_feed_dict

tf.compat.v1.disable_eager_execution()

npz_path = 'F:/ProjectData/mesh_feature/test/test_npz/front'
log_dir = 'F:/ProjectData/mesh_feature/test/log_dir'
batch_size = 1

plc, input_names = build_plc(BLOCK_NUM)
model = ExtractModel(CHANNELS, coarse_level=C_LEVEL, fc_dim=4)
load_time_dir = '20191031-1754/rutine'  # where to restore the model
ckpt_file = 'ckpt-413'
output = model(plc, need_sqeeze=False)
output = tf.identity(output, 'output_node')
ckpt_full_dir = os.path.join(CKPT_PATH, load_time_dir)
ckpt_full_path = os.path.join(ckpt_full_dir, ckpt_file)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(ckpt_full_path)

need_freeze = False
need_infer = True
need_summary = True

version = 1
model_path = os.path.join('../freeze_output', str(version))

# %%

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
status.initialize_or_restore(sess)
# for node in tf.train.list_variables(tf.train.latest_checkpoint(ckpt_full_dir)):
#     print(node)
if need_freeze:
    if os.path.exists(model_path):
        import shutil
        
        shutil.rmtree(model_path)
    else:
        os.mkdir(model_path)
    
    
    def build_input_info(plc_dict):
        tensor_infos = {'vertice': plc['vertice']}
        adj_infos = {'adj_%d' % i: adj_plc for i, adj_plc in enumerate(plc['adjs'])}
        perm_infos = {'perm_%d' % i: perm_plc for i, perm_plc in enumerate(plc['perms'])}
        tensor_infos.update(adj_infos)
        tensor_infos.update(perm_infos)
        return tensor_infos
    
    
    # ordinary model
    tf.compat.v1.saved_model.simple_save(sess, model_path, build_input_info(plc), {'output_node': output})
    write_pb(input_saved_model_dir=model_path,
             output_graph_filename="../output_graph.pb")

elif need_infer:
    data_gen = Data_Gen('F:/ProjectData/mesh_feature/Case_npz/back')
    rf = Rotate_feed(
        rot_num=2,
        rot_range=[np.pi / 18., np.pi / 18., np.pi / 18.],
        angle_fixed=True,
        data_gen=data_gen
    )
    epoch_end = False
    
    # while (not epoch_end):
    feed_numpy, epoch_end = rf.rotate_case()
    x = feed_numpy['vertice'] #[b,n,3]
    adjs = feed_numpy['adjs'] #[[n,K]]
    perms = feed_numpy['perms'] #[[n]]
    mask = feed_numpy['mask'] #[[n]]
    
    feed_dict = build_feed_dict(plc, x, adjs, perms)
    result = sess.run(output, feed_dict=feed_dict)
    output = np.reshape(result, [-1, FEAT_CAP, 3])
    result = output[:,mask,:]  #[1,f_num,3]
    print(result)
    sess.close()

    # %%
    
    # Camera and scene configuration.
    config_dict = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 1,
            # 'color':0x000fff
        }
    }
        
        # Read all sample PLY files.
        
        
        
        # data_gen = Data_Gen(npz_path)
        # data, npz_name, epoch_end = data_gen.load_pkg()
        
        # vertices=data['x'].astype(np.float32)
        # features=data['y'].astype(np.float32)
        #
        # # Add batch dimension, so our data will be of shape BxNxC.
        # points_v = vertices
        # points_f=features[:,1:]
        
        # vertices = data['x'][1].astype(np.float32)
        # features = data['y'][1].astype(np.float32)
        
        # Add batch dimension, so our data will be of shape BxNxC.
    points_v = x
    points_f = result
    points = np.concatenate([points_v, points_f], axis=1)
    
    colors_v = np.ones_like(points_v) * [0, 0, 255]
    colors_f = np.ones_like(points_f) * [0, 255, 0]
    colors = np.concatenate([colors_v, colors_f], axis=1)
        
        # %%
    tf.compat.v1.disable_eager_execution()

    points_tensor = tf.compat.v1.placeholder(tf.float32, points.shape)
    colors_tensor = tf.compat.v1.placeholder(tf.int32, colors.shape)
    
    summary = mesh_summary.op(
        'v_color_tensor',
        vertices=points_tensor,
        colors=colors_tensor,
        config_dict=config_dict
    )
    
    # Create summary writer and session.
    writer = tf.compat.v1.summary.FileWriter(log_dir)
    sess = tf.compat.v1.Session()
    
    # %%
    
    summaries = sess.run([summary], feed_dict={
        points_tensor: points,
        colors_tensor: colors,
    })
    # Save summaries.
    for summary in summaries:
        writer.add_summary(summary)

# %%


