import os


from Direction.src.config import *
from Direction.src.dire_data import process_data
from common.freeze_wrapper import write_pb
from common.plc import *
from common.extract_model import ExtractModel

tf.compat.v1.disable_eager_execution()

plc,input_names=build_plc(BLOCK_NUM,adj_dim=ADJ_K,need_batch=True)

# optimizer = tf.train.AdamOptimizer() #1.x
model=ExtractModel(CHANNELS,coarse_level=C_LEVEL,fc_dim=4)

# load_time_dir = '20190821-1059/rutine'  # where to restore the model
# ckpt_file = 'ckpt-160'
load_time_dir = '20190823-1511/rutine'  # where to restore the model
ckpt_file = 'ckpt-80'


output = model(plc,need_sqeeze=True)
output=tf.identity(output,'output_node')

ckpt_full_dir = os.path.join(CKPT_PATH, load_time_dir)
ckpt_full_path = os.path.join(ckpt_full_dir, ckpt_file)
checkpoint = tf.train.Checkpoint(model=model)
# pass if ckpt is equal to the objects
# optimizer is not built, but it's in the ckpt, so this assert won't pass
status = checkpoint.restore(ckpt_full_path)

# pass if ckpt is equal to or larger than objects
# all the sub objects can access to their corresponding value in ckpt
status.assert_existing_objects_matched()


data_path = "F:/ProjectData/mesh_direction/2aitest/low"

need_freeze=True
version=1
model_path=os.path.join('../freeze_output',str(version))
with tf.compat.v1.Session() as sess:
    status.initialize_or_restore(sess)
    # for node in tf.train.list_variables(tf.train.latest_checkpoint(ckpt_full_dir)):
    #     print(node)
    if need_freeze:
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
        else:
            os.mkdir(model_path)

        def build_input_info(plc_dict):
            tensor_infos={'vertice':plc['vertice']}
            adj_infos={'adj_%d' % i:adj_plc for i,adj_plc in enumerate(plc['adjs'])}
            perm_infos={'perm_%d' % i:perm_plc for i,perm_plc in enumerate(plc['perms'])}
            tensor_infos.update(adj_infos)
            tensor_infos.update(perm_infos)
            return tensor_infos
        # ordinary model
        tf.compat.v1.saved_model.simple_save(sess, model_path, build_input_info(plc), {'output_node': output})
        
# ############# SERVER MODEL
#
#         export_path = os.path.join(
#             tf.compat.as_bytes('../freeze_output'),
#             tf.compat.as_bytes(str(1)))
#         print('Exporting trained model to', export_path)
#         builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
#
#         def build_input_info(plc_dict):
#             tensor_infos={'vertice':tf.compat.v1.saved_model.utils.build_tensor_info(plc['vertice'])}
#             adj_infos={'adj_%d' % i:tf.compat.v1.saved_model.utils.build_tensor_info(adj_plc) for i,adj_plc in enumerate(plc['adjs'])}
#             perm_infos={'perm_%d' % i:tf.compat.v1.saved_model.utils.build_tensor_info(perm_plc) for i,perm_plc in enumerate(plc['perms'])}
#             tensor_infos.update(adj_infos)
#             tensor_infos.update(perm_infos)
#             return tensor_infos
#
#
#         inputs_info=build_input_info(plc)
#         output_info = {'output': tf.compat.v1.saved_model.utils.build_tensor_info(output)}
#
#         prediction_signature = (
#             tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
#                 inputs=inputs_info,
#                 outputs=output_info,
#                 method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))
#
#         builder.add_meta_graph_and_variables(
#             sess, [tf.saved_model.SERVING],
#             signature_def_map={
#                 'predict_direction':
#                     prediction_signature
#             },
#             main_op=tf.compat.v1.tables_initializer(),
#             strip_default_attrs=True)
#         builder.save()
#         # ##################  SERVER MODEL
        
        
        
        write_pb(input_saved_model_dir=model_path,
                 output_graph_filename="../output_graph.pb")





    else:
        X, Adjs, Perms=process_data(data_path, 'case_test.txt')
        for x,adjs,perms in zip(X, Adjs, Perms):
            feed_dict=build_feed_dict(plc,x,adjs,perms)
            result=sess.run(output,feed_dict=feed_dict)
            print(result)
    

