import os

import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.python.tools import optimize_for_inference_lib

from common.combine import gen_frozen_graph, input_path, output_node_names


def write_pb(sess, time_dir, ckpt_file,input_names,input_types,need_optimize=False):
    gd = sess.graph.as_graph_def()
    
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
    constant_graph = graph_util.convert_variables_to_constants(sess, gd, [output_node_names])
    graph_io.write_graph(constant_graph, time_dir, input_path, as_text=False)

    # tf.train.write_graph(constant_graph, time_dir, input_path)
    # gen_frozen_graph( time_dir,ckpt_file)
    freeze_graph()

    if need_optimize:
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(os.path.join(time_dir, 'output_graph.pb'), "rb") as f:
            data = f.read()
        input_graph_def.ParseFromString(data)
        
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            # ["input", "image_shape", "phase"],  # an array of the input node(s)
            input_names,
            ["output_node"],  # an array of output nodes
            #[tf.float32.as_datatype_enum, tf.float32.as_datatype_enum, tf.bool.as_datatype_enum]
            input_types
        )
        
        # Save the optimized graph
        
        f = tf.gfile.FastGFile(os.path.join(time_dir, 'optimized_graph.pb'), "w")
        f.write(output_graph_def.SerializeToString())
