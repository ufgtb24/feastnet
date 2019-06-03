import tensorflow as tf

from Direction.src.config import CHANNELS
from Direction.src.dire_data import Data_Gen, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model_keras import DirectionModel

keras=tf.keras

optimizer=keras.optimizers.Adam(1.e-3)
model=DirectionModel(CHANNELS,fc_dim=4)
data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')
rf=Rotate_feed(10,data_gen)

for epoch in range(3):
    epoch_end=False
    idx=0
    while(not epoch_end):
        feed_dict,epoch_end = rf.get_feed()
        with tf.GradientTape() as tape:
            
            output = model(feed_dict)
            output = tf.reshape(output, [4])
            loss = pose_estimation_loss(feed_dict['input'], feed_dict['label'], output)
            print(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())
        if idx%10==0:
            print(loss)
