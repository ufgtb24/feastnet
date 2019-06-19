import tensorflow as tf
from common.custom_layer import Conv_Mesh

class Block(tf.keras.layers.Layer):
    def __init__(self, ch_in,ch_out,coarse_level):
        self.conv1=Conv_Mesh(ch_in, 9)
        self.conv2=Conv_Mesh(ch_out, 9)
        self.Pool_layers=[tf.keras.layers.AveragePooling1D(pool_size=2,strides=2)]*coarse_level
        super(Block, self).__init__()
        
    def call(self,x,adj,perm):
        '''
        
        :param x: B,N,3
        :param adj: N,K
        :param perm:
        :return:
        '''
        #[B,N,C1]
        net=tf.nn.relu(self.conv1(x,adj))
        # net=tf.nn.relu(self.conv3(net,adj))
        #[B,N,C2]
        net=tf.nn.relu(self.conv2(net,adj))
        if perm is not None:
            # [B,M,C]
            net = self.perm_data(net, perm)
            for pool in self.Pool_layers:
                net=pool(net)
        # [B,M/(2**CL),C]
        return net
    
    def perm_data(self, input, indices):
        """
        排列成待卷积的序列
        Permute data matrix, i.e. exchange node ids,
        so that binary unions form the clustering tree.
        new x can be used for pooling
        :param input:  B, N,channel
        :param indices: M
        :return: B, M(M>N), channel
        """
        #[B,M-N,C]
        fake_node = tf.zeros(
            [tf.shape(input)[0],tf.shape(indices)[0] - tf.shape(input)[1], tf.shape(input)[2]],
            dtype=tf.float32)
        #[B,M,C]
        sample_array = tf.concat([input, fake_node], axis=1)
        #[B,M,C]
        perm_data = tf.gather(sample_array, indices,axis=1)
        return perm_data
    


class DirectionModel(tf.keras.Model):
    def __init__(self,block_CHL, coarse_level,fc_dim):
        super(DirectionModel, self).__init__()
        self.FC_output=tf.keras.layers.Dense(fc_dim)
        self.block_num=len(block_CHL)-1
        self.Blocks=[]
        # self.block_1,self.block_2,self.block_3
        for idx, (ch_in, ch_out) in enumerate(zip(block_CHL[:-1], block_CHL[1:])):  # 6
            # exec('self.block_%d = Block(ch_in, ch_out, coarse_level=coarse_level)'%idx)
            # exec('self.Blocks.append(self.block_%d)'%idx)
            # this kind of sub layers can not be recorded by tf.train.checkpoint, for its lack of key
            self.Blocks.append(Block(ch_in, ch_out, coarse_level=coarse_level))

    def call(self, feed_dict):
        """Run the model. will call build
        """
        
        # [B,N_INPUT,C]
        net=feed_dict['vertice']
        adjs=feed_dict['adjs']
        perms=feed_dict['perms']

        for idx,block in enumerate(self.Blocks):
            if idx == self.block_num-1:
                # [B,N_FINAL,C_FINAL]
                net = block(net,adjs[idx],None)
            else:
                # [B,N_INNER,C_INNER]
                net = block(net,adjs[idx],perms[idx])
        # [B,C_FINAL]
        net = tf.reduce_mean(net, axis=1)
        # [B,OUTPUT_DIM]
        net=self.FC_output(net)
        return net


