import tensorflow as tf
from common.custom_layer import Conv_Mesh

class Block(tf.keras.layers.Layer):
    def __init__(self, ch_in,ch_out,coarse_level, **kwargs):
        self.conv1=Conv_Mesh(ch_in, 9)
        self.conv2=Conv_Mesh(ch_out, 9)
        self.Pool_layers=[tf.keras.layers.MaxPooling1D(pool_size=2,strides=2)]*coarse_level
        super(Block, self).__init__(**kwargs)
    
    def call(self,x,adj,perm,need_pool=True):
        
        net=tf.nn.relu(self.conv1(x,adj))
        net=tf.nn.relu(self.conv2(net,adj))
        if need_pool:
            net = self.perm_data(net, perm)
            net=tf.expand_dims(net,axis=0)
            for pool in self.Pool_layers:
                net=pool(net)
            net=tf.squeeze(net,axis=0)
        return net
    
    def perm_data(self, input, indices):
        """
        排列成待卷积的序列
        Permute data matrix, i.e. exchange node ids,
        so that binary unions form the clustering tree.
        new x can be used for pooling
        :param input:  M,channel
        :param indices: Mnew
        :return: Mnew, channel
        """
        
        fake_node = tf.zeros([indices.shape[0] - tf.shape(input)[0], input.shape[1]], dtype=tf.float32)
        sample_array = tf.concat([input, fake_node], axis=0)  # [Mnew,channel]
        perm_data = tf.gather(sample_array, indices)
        return perm_data
    


class DirectionModel(tf.keras.Model):
    def __init__(self,block_CHL, fc_dim):
        super(DirectionModel, self).__init__()
        self.FC_output=tf.keras.layers.Dense(fc_dim)
        self.block_num=len(block_CHL)-1
        self.Blocks=[]
        
        for idx, (ch_in, ch_out) in enumerate(zip(block_CHL[:-1], block_CHL[1:])):  # 6
            self.Blocks.append(Block(ch_in, ch_out, coarse_level=2))
    

    def call(self, feed_dict):
        """Run the model."""
        
        net=feed_dict['input']
        adjs=feed_dict['adjs']
        perms=feed_dict['perms']

        for idx,block in enumerate(self.Blocks):
            if idx == self.block_num-1:
                net = block(net,adjs[idx],perms[idx],need_pool=False)
            else:
                net = block(net,adjs[idx],perms[idx],need_pool=True)

        net = tf.reduce_mean(net, axis=0)  # [512]
        net = tf.expand_dims(net, axis=0)
        net=self.FC_output(net)

        return net


