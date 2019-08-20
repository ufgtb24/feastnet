from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
output_node_names = "output_node"
input_binary = False
input_saver_def_path = False
restore_op_name = None
filename_tensor_name = None
clear_devices = False
input_meta_graph = False
checkpoint_path = None
input_graph_filename = None
saved_model_tags = tag_constants.SERVING

def write_pb(input_saved_model_dir,output_graph_filename):
    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                            input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                            saved_model_tags)


def load_graph(sess,frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v1.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with sess.graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return sess.graph