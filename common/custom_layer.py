import tensorflow as tf






class Conv_Mesh(tf.keras.layers.Layer):

  def __init__(self, out_channels, M,ring=1,**kwargs):
    self.out_channels = out_channels
    self.M=M
    self.ring=ring
    super(Conv_Mesh, self).__init__(**kwargs)

  def build(self, input_shape):
    self.in_channels = input_shape[1]

    # Create a trainable weight variable for this layer.
    self.W = self.add_variable(name='W',
                                  shape=[self.M,self.out_channels,self.in_channels],
                                  initializer='uniform',
                                  trainable=True)
    self.b=self.add_variable(name='b',shape=[self.out_channels],
                             initializer='uniform',
                             trainable=True)
    self.u=self.add_variable(name='u',shape=[self.M,self.in_channels],
                             initializer='uniform',
                             trainable=True)
    self.c=self.add_variable(name='c',shape=[self.M],
                             initializer='uniform',
                             trainable=True)
    
    # Make sure to call the `build` method at the end
    super(Conv_Mesh, self).build(input_shape)

  def get_patches_2(self,x, adj):
    '''
    获得 x 的adj patch
    :param x: num_points, in_channels
    :param adj: num_points, K
    :return:
    '''
    num_points, in_channels = x.get_shape().as_list()
    input_size, K = adj.shape
    zeros = tf.zeros([1, in_channels], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    x = tf.concat([zeros, x], 0)  # [num_points+1, in_channels]
  
    ############### 2-ring
    zeros_adj = tf.zeros([1, K], dtype=tf.int32)
    _adj = tf.concat([zeros_adj, adj], 0)  # [num_points+1 , K]
  
    patches_adj = tf.gather(_adj, adj)  # [num_points,K,K]
    K2 = K * K
    patches_adj = tf.reshape(patches_adj, [-1, K2])
  
    def cut_nodes(i, x):
      y, idx = tf.unique(x)
      z = tf.boolean_mask(y, tf.not_equal(y, i))
      paddings = [[0, K2 - tf.shape(z)[0]]]
      z = tf.pad(z, paddings)
      return z
  
    # [num_points,K2] adj_2 need to save in numpy
    adj_2 = tf.map_fn(lambda x: cut_nodes(x[0], x[1]),
                      (tf.range(patches_adj.shape[0]) + 1, patches_adj),
                      dtype=tf.int32)
    patches = tf.gather(x, adj_2)  # [num_points,K2,in_channels]
  
    return patches

  def get_patches_1(self,x, adj):
    '''
    获得 x 的adj patch
    :param x:  N, C
    :param adj:  N, K
    :return: N,K,C
    '''
    num_points, in_channels = x.shape
    zeros = tf.zeros([1, in_channels], dtype=tf.float32)
    # 索引为0的邻接点，会索引到 0,0
    x = tf.concat([zeros, x], 0)  # [N+1, C]
    patches = tf.gather(x, adj)  # [N,K,C]
    return patches

  def get_patches(self,x, adj, ring_num):
    K = tf.shape(adj)[1]
    if ring_num == 1:
      return K, self.get_patches_1(x, adj)
    elif ring_num == 2:
      return K * K, self.get_patches_2(x, adj)

  def get_weight_assigments(self,x,adj,u,c,ring):
    M, in_channels = tf.shape(u)
    # [N, K, ch]
    K, patches = self.get_patches(x, adj, ring)
    # [ N, ch, 1]
    x = tf.reshape(x, [-1, in_channels, 1])
    # [ N, ch, K]
    patches = tf.transpose(patches, [0, 2, 1])
    # [N, ch, K]
    patches = tf.subtract(x, patches)  # invariance
    # [ch, N, K]
    patches = tf.transpose(patches, [1, 0, 2])
    # [ch, N*K]
    x_patches = tf.reshape(patches, [in_channels, -1])
    #  M, N*K  = [M, ch]  x  [ch, N*K]
    patches = tf.matmul(u, x_patches)
    # M, N, K
    patches = tf.reshape(patches, [M, -1, K])
    # [K, N, M]
    patches = tf.transpose(patches, [2, 1, 0])
    # [K, N, M]
    patches = tf.add(patches, c)
    # N, K, M
    patches = tf.transpose(patches, [1, 0, 2])
    patches = tf.nn.softmax(patches)
    return patches

    
    
    
  def call(self,x,adj):
    # Calculate neighbourhood size for each input - [N, neighbours]
    adj_size = tf.count_nonzero(adj, 1)  # [N] 每个元素 是该点的邻接点数量
    # deal with unconnected points: replace NaN with 0
    non_zeros = tf.not_equal(adj_size, 0)  # [N] bool  是否有孤立点
    adj_size = tf.cast(adj_size, tf.float32)
    adj_size = tf.where(non_zeros, tf.reciprocal(adj_size), tf.zeros_like(adj_size))  # 非孤立点 删选出来
    # [N, 1, 1]
    adj_size = tf.reshape(adj_size, [-1, 1, 1])
    # [N, K, M] 当K index 到 0 时， M 维相等
    q = self.get_weight_assigments(x, adj, self.u, self.c, self.ring)
    # [C, N]
    x = tf.transpose(x, [1, 0])
    # [M*O,C]
    W = tf.reshape(self.W, [self.M * self.out_channels, self.in_channels])  # 卷积核参数
    # [M*O, N]
    wx = tf.matmul(W, x)  # 卷积
    # [N, M*O]
    wx = tf.transpose(wx, [1, 0])
    # adj中k为0的索引取到的 wx 也为0
    # [N, K, M*O]
    K, patches = self.get_patches(wx, adj, self.ring)
    # [ N, K, M, O]
    patches = tf.reshape(patches, [-1, K, self.M, self.out_channels])
    # [O, N, K, M]
    patches = tf.transpose(patches, [3, 0, 1, 2])
    # [N, K, M]*[O, N, K, M]=[O, N, K, M] element-wise
    patches = tf.multiply(q, patches)
    # [N, K, M, O]
    patches = tf.transpose(patches, [1, 2, 3, 0])
    # Add all the elements for all neighbours for a particular m
    # sum data in index K is zero, so need 归一化
    patches = tf.reduce_sum(patches, axis=1)  # [ N, M, O]
    # [N, 1, 1]*[ N, M, O]=[ N, M, O]
    patches = tf.multiply(adj_size, patches)  # /Ni
    # Add add elements for all m， 因为不是 reduce_mean 所以支持M中的0
    # [ N, O]
    patches = tf.reduce_sum(patches, axis=1)
    # [N, O]
    patches = patches + self.b
    return patches

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(Conv_Mesh, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
  
  
  