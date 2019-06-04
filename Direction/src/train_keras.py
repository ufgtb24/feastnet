import tensorflow as tf

from Direction.src.config import CHANNELS
from Direction.src.dire_data import Data_Gen, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model_keras import DirectionModel
tf.enable_eager_execution()

print(tf.executing_eagerly())

keras=tf.keras
optimizer = tf.train.AdamOptimizer()
# optimizer = keras.optimizers.Adam(1e-5)  # 2.0
model=DirectionModel(CHANNELS,fc_dim=4)
# data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')
rf=Rotate_feed(10,data_gen)

for epoch in range(10000):
    # print("epoch ",epoch)
    epoch_end=False
    
    while(not epoch_end):
        feed_dict,epoch_end = rf.get_feed()
        with tf.GradientTape() as tape:
            
            output = model(feed_dict)
            output = tf.reshape(output, [4])
            loss = pose_estimation_loss(feed_dict['input'], feed_dict['label'], output)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    print("epoch %d  : %f"%(epoch,loss))
