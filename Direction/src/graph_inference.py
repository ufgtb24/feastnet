from Direction.src.dire_data import process_data
from common.freeze_wrapper import load_graph
from Direction.src.config import *
from common.plc import *

print(tf.__version__)
# adj_path='../adj.txt'
# perms, adjs = multi_coarsen(adj_path, ADJ_K, BLOCK_NUM - 1, C_LEVEL)

data_path = "F:/ProjectData/mesh_direction/2aitest/low"
X, Adjs, Perms = process_data(data_path, 'case_test.txt')

with tf.compat.v1.Session() as sess:
    load_graph(sess, "../output_graph.pb")
    names=[n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    for name in names:
        print(name)
    pred_end = sess.graph.get_tensor_by_name('import/output_node:0')



    for x, adjs, perms in zip(X, Adjs, Perms):
        feed_dict = build_feed_dict_pb(BLOCK_NUM, x, adjs, perms)
        result = sess.run(pred_end, feed_dict=feed_dict)
        print(result)

