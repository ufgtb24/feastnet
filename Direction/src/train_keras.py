import tensorflow as tf

from Direction.src.config import CHANNELS
from Direction.src.dire_data import Data_Gen, Rotate_feed
from Direction.src.loss import pose_estimation_loss
from common.model_keras import DirectionModel
# tf.enable_eager_execution()
# tf.debugging.set_log_device_placement(True)
# print(tf.executing_eagerly())

keras=tf.keras
# optimizer = tf.train.AdamOptimizer()
optimizer = keras.optimizers.Adam()
model=DirectionModel(CHANNELS,fc_dim=4)
# data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')
rf=Rotate_feed(30,data_gen)
mean_metric = keras.metrics.Mean()
for epoch in range(1000):
    # print("epoch ",epoch)
    mean_metric.reset_states()
    epoch_end=False
    while(not epoch_end):
        feed_dict,epoch_end = rf.get_feed()
        with tf.GradientTape() as tape:
            
            output = model(feed_dict)
            output = tf.reshape(output, [4])
            loss = pose_estimation_loss(feed_dict['input'], feed_dict['label'], output)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        mean_metric.update_state(loss)
    print(loss)
    print("epoch %d  : %f"%(epoch,mean_metric.result().numpy()))
    model.save_weights('/home/yu/PycharmProjects/feastnet/Direction/data/',
                       save_format='tf')
