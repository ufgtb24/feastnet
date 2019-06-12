from Direction.src.dire_data import Data_Gen, Rotate_feed
import tensorflow as tf
import trimesh
import tensorboard

from tensorboard.plugins.mesh import summary as mesh_summary
tf.enable_eager_execution()

print(tf.__version__)
print(tf.executing_eagerly())

# data_gen = Data_Gen('F:/ProjectData/mesh_direction/2aitest/low/npz')
data_gen = Data_Gen('/home/yu/Documents/project_data/low/npz')

rf=Rotate_feed(10,data_gen)
config_dict = {
    'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
    'lights': [
        {
          'cls': 'AmbientLight',
          'color': '#ffffff',
          'intensity': 0.75,
        }, {
          'cls': 'DirectionalLight',
          'color': '#ffffff',
          'intensity': 0.75,
          'position': [0, -1, 2],
        }],
    'material': {
      'cls': 'MeshStandardMaterial',
      'roughness': 1,
      'metalness': 0
    }
}
log_dir = '/home/yu/PycharmProjects/feastnet/Direction/data'
# vertices_tensor = tf.placeholder(tf.float32, [1,None,3])
sess = tf.Session()
for i in range(3):
    feed_dict, epoch_end = rf.get_feed()
    vertices_tensor=tf.expand_dims(feed_dict['input'],0)
    print(vertices_tensor.shape)
    meshes_summary = mesh_summary.op(
        'mesh_color_tensor', vertices=vertices_tensor, config_dict=config_dict)
    writer = tf.summary.FileWriter(log_dir)
    writer.add_summary(meshes_summary)
    
    
