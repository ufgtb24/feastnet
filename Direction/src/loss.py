import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion

def pose_estimation_loss(ori_vertices,y_true, y_pred):
  """Pose estimation loss used for training.

  This loss measures the average of squared distance between some vertices
  of the mesh in 'rest pose' and the transformed mesh to which the predicted
  inverse pose is applied. Comparing this loss with a regular L2 loss on the
  quaternion and translation values is left as exercise to the interested
  reader.

  Args:
    y_true: The ground-truth value. [n,c]
    y_pred: The prediction we want to evaluate the loss for. [b,4]

  Returns:
    A scalar value containing the loss described in the description above.
  """

  # vertices.shape: (num_vertices, 3)
  # corners.shape:(num_vertices, 1, 3)
  corners = tf.expand_dims(ori_vertices, axis=1)
  # corners = ori_vertices

  # transformed_corners.shape: (num_vertices, batch, 3)
  # q and t shapes get pre-pre-padded with 1's following standard broadcast rules.
  transformed_corners = quaternion.rotate(corners, y_pred)

  # recovered_corners.shape: (num_vertices, batch, 3)
  recovered_corners = quaternion.rotate(transformed_corners ,
                                        quaternion.inverse(y_true))

  # vertex_error.shape: (num_vertices, batch)
  vertex_error = tf.reduce_sum((recovered_corners - corners)**2, axis=-1)

  return tf.reduce_mean(vertex_error)