import tensorflow as tf
import numpy as np
def build_plc(block_num,label_shape, adj_dim):
    adjs = []
    perms = []
    input = tf.placeholder(tf.float32, [None, 3])
    label = tf.placeholder(tf.float32, label_shape)
    for i in range(block_num):
        adjs.append(tf.placeholder(tf.int32, [None, adj_dim]))
        perms.append(tf.placeholder(tf.int32, [None]))
    return {'input': input, 'label': label, 'adjs': adjs, 'perms': perms}


def build_feed_dict(plc, data, iter):
    feed_dict = {
        plc['input']: data['x'][iter],
        plc['label']: data['y'][iter],
    }
    
    adjs_dict = {adj_plc: data['adj'][iter][idx] for idx, adj_plc in enumerate(plc['adjs'])}
    perms_dict = {perm_plc: data['perm'][iter][idx] for idx, perm_plc in enumerate(plc['perms'])}
    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    return feed_dict


def build_plc_b(block_num, adj_dim):
    adjs = []
    perms = []
    input = tf.placeholder(tf.float32, [1,None, 3])
    for i in range(block_num):
        adjs.append(tf.placeholder(tf.int32, [None, adj_dim]))
        if i!=block_num-1:
            perms.append(tf.placeholder(tf.int32, [None]))
    return {'input': input, 'adjs': adjs, 'perms': perms}


def build_feed_dict_b(plc, input,adjs,perms):
    feed_dict = {
        plc['input']: input[np.newaxis,:],
    }
    
    adjs_dict = {adj_plc: adjs[idx] for idx, adj_plc in enumerate(plc['adjs'])}
    perms_dict = {perm_plc: perms[idx] for idx, perm_plc in enumerate(plc['perms'])}
    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    return feed_dict



