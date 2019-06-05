import tensorflow as tf
def build_plc(block_num,label_shape, adj_dim):
    adjs = []
    perms = []
    input = tf.placeholder(tf.float32, [None, 3])
    label = tf.placeholder(tf.float32, label_shape)
    rot_iter = tf.placeholder(tf.int32)
    
    for i in range(block_num):
        adjs.append(tf.placeholder(tf.int32, [None, adj_dim]))
    for i in range(block_num-1):
        perms.append(tf.placeholder(tf.int32, [None]))
    return {'input': input, 'label': label, 'rot_iter':rot_iter,
            'adjs': adjs, 'perms': perms}


def build_feed_dict(plc, data,rot_iter):
    feed_dict = {
        plc['input']: data['x'],
        plc['label']: data['y'],
    }
    
    adjs_dict = {adj_plc: data['adjs'][idx] for idx, adj_plc in enumerate(plc['adjs'])}
    perms_dict = {perm_plc: data['perms'][idx] for idx, perm_plc in enumerate(plc['perms'])}
    rot_dict = {plc['rot_iter']: rot_iter}

    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    feed_dict.update(rot_dict)
    return feed_dict



