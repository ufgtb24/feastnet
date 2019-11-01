import tensorflow as tf
import numpy as np


def build_plc(block_num ,need_batch=False):
    adjs = []
    perms = []
    input_names = {}
    vertice = tf.compat.v1.placeholder(tf.float32, [1, None, 3], name='vertice')
    input_names['vertice'] = vertice
    
    if need_batch:
        adjs_shape=[1,None, None]
        perms_shape=[1,None]
    else:
        adjs_shape=[None, None]
        perms_shape=[None]

    for i in range(block_num):
        adj_plc = tf.compat.v1.placeholder(tf.int32, adjs_shape, name='adj_%d' % i)
        adjs.append(adj_plc)
        input_names['adj_%d' % i] = adj_plc
        if i != block_num - 1:
            perm_plc = tf.compat.v1.placeholder(tf.int32, perms_shape, name='perm_%d' % i)
            perms.append(perm_plc)
            input_names['perm_%d' % i] = perm_plc
    plc = {'vertice': vertice, 'adjs': adjs, 'perms': perms}
    return plc, input_names


def build_feed_dict(plc, vertice, adjs, perms):
    '''
    dict {placeholders:input array}
    '''
    feed_dict = {
        plc['vertice']: vertice[np.newaxis, :],
    }
    
    adjs_dict = {adj_plc: adjs[idx] for idx, adj_plc in enumerate(plc['adjs'])}
    perms_dict = {perm_plc: perms[idx] for idx, perm_plc in enumerate(plc['perms'])}
    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    return feed_dict

def build_feed_dict_pb(block_num, vertice, adjs, perms):
    '''
    dict {placeholders_name :input array}
    '''

    feed_dict = {'import/vertice:0': vertice[np.newaxis, :]}
    for i in range(block_num):
        feed_dict['import/adj_%d:0' % i] = adjs[i]
        if i != block_num - 1:
            feed_dict['import/perm_%d:0' % i] = perms[i]
    return feed_dict

