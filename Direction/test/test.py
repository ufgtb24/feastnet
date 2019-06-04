import tensorflow as tf

tf.enable_eager_execution()
a=tf.constant(0,tf.float32,(5,2))
print(tf.shape(a)[0])