import tensorflow as tf

def loss_func(pred,label):
    '''
    
    :param pred: [feature_cap,3]
    :param label: [feature_cap,4]
    :return:
    '''
    mask=tf.cast(label[:,0],tf.bool) #[feature_cap]
    label=label[:,1:] #[feature_cap,3]
    mask_label=tf.boolean_mask(label,mask)  #[valid_feat_num,3]
    mask_pred=tf.boolean_mask(pred,mask) #[valid_feat_num,3]
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(mask_label-mask_pred),axis=-1))
    return loss