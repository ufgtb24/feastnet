import os

import tensorflow as tf

from Direction.src.dire_data import process_data
from Direction.src.freeze_wrapper import load_graph
from Direction.src.config import *
from Direction.src.plc import *
from common.coarsening import multi_coarsen
print(tf.__version__)
# adj_path='../adj.txt'
# perms, adjs = multi_coarsen(adj_path, ADJ_K, BLOCK_NUM - 1, C_LEVEL)
tf.compat.v1.disable_eager_execution()

version=1
model_path=os.path.join('../pb_save',str(version))

with tf.compat.v1.Session() as sess:
    load_graph(sess, "../output_graph.pb")

    # names=[n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    # for name in names:
    #     print(name)

    if os.path.exists(model_path):
        import shutil
    
        shutil.rmtree(model_path)
    else:
        os.mkdir(model_path)


    def build_input_info():
        tensor_infos = {'vertice': sess.graph.get_tensor_by_name('import/vertice:0')}
        adj_infos = {'adj_%d' % i: sess.graph.get_tensor_by_name('import/adj_%d:0'%i) for i in range(3)}
        perm_infos = {'perm_%d' % i: sess.graph.get_tensor_by_name('import/perm_%d:0'%i) for i in range(2)}
        tensor_infos.update(adj_infos)
        tensor_infos.update(perm_infos)
        return tensor_infos



    pred_end = sess.graph.get_tensor_by_name('import/output_node:0')
    data_path = "F:/ProjectData/mesh_direction/2aitest/low"
    X, Adjs, Perms = process_data(data_path, 'case_test.txt')

    for x, adjs, perms in zip(X, Adjs, Perms):
        feed_dict = build_feed_dict_pb(BLOCK_NUM, x, adjs, perms)
        result = sess.run(pred_end, feed_dict=feed_dict)
        print(result)
    # ordinary model
    tf.compat.v1.saved_model.simple_save(sess, model_path, build_input_info(), {'output_node': pred_end})





