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
model_path=os.path.join('../freeze_output',str(version))
plc,input_names=build_plc(BLOCK_NUM,adj_dim=ADJ_K)

with tf.compat.v1.Session() as sess:
    load_graph(sess, "../output_graph.pb")
    pred_end = sess.graph.get_tensor_by_name('import/output_node:0')

    # names=[n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    # for name in names:
    #     print(name)

    if os.path.exists(model_path):
        import shutil
    
        shutil.rmtree(model_path)
    else:
        os.mkdir(model_path)


    def build_input_info():
        tensor_infos = {'vertice': sess.graph.get_tensor_by_name('import/output_node:0')}
        adj_infos = {'adj_%d' % i: sess.graph.get_tensor_by_name('import/adj_%d:'%i) for i in range(3)}
        perm_infos = {'perm_%d' % i: sess.graph.get_tensor_by_name('import/perm_%d:'%i) for i in range(2)}
        tensor_infos.update(adj_infos)
        tensor_infos.update(perm_infos)
        return tensor_infos



    pred_end = sess.graph.get_tensor_by_name('import/output_node:0')

    # ordinary model
    tf.compat.v1.saved_model.simple_save(sess, model_path, build_input_info(), {'output_node': pred_end})





