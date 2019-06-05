import tensorflow as tf
class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    a=tf.constant(100)
    for i in range(5):
      a+=1
    return a


model = CustomModel()

print(model(tf.constant([-2, -4])))