import tensorflow as tf

def loss_func(pred,label,mask):
    '''
    
    :param pred: [rot_num,feature_cap,3]
    :param label: [rot_num,feature_cap,3]
    :param mask: [feature_cap]
    :return:
    '''
    mask_label=tf.boolean_mask(label,mask,axis=1)  #[rot_num,valid_feat_num,3]
    mask_pred=tf.boolean_mask(pred,mask,axis=1) #[rot_num,valid_feat_num,3]
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(mask_label-mask_pred),axis=-1))
    return loss