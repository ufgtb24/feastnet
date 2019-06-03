from Direction.src.dire_data import Data_Gen, Rotate_feed
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
# tf.enable_eager_execution()

data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')
rf=Rotate_feed(10,data_gen)
idx=0
while(True):
    feed_dict, epoch_end = rf.get_feed()
    if epoch_end:
        print(idx)
    idx+=1