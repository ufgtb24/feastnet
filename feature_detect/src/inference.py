import os
import time

import tensorflow as tf
import numpy as np

from common.place_holder_ops import build_plc, build_feed_dict
from feature_detect.src.feat_data import Data_Gen
from fill_holes.src.data_process import get_training_data
from common.model import *
from feature_detect.src.config import *

data_path = 'F:/ProjectData/surface/leg/valid'
plc=build_plc(BLOCK_NUM,label_shape=[FEAT_CAP,4],adj_dim=ADJ_K)
# [1,FEAT_CAP*3]
output = Mesh2FC(plc, CHANNELS, fc_dim=FEAT_CAP*3)
output=tf.squeeze(output,axis=0)
saver = tf.train.Saver()

# NUM_PARALLEL_EXEC_UNITS=4
# config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
#                         inter_op_parallelism_threads=2,
#                         allow_soft_placement=True,
#                         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
#
# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["KMP_BLOCKTIME"] = "0"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

def wirte_result(root_path,case_list,output_list):
    result_path=os.path.join(root_path,'result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(os.path.join(root_path,case_list))as f:
        names=f.read().splitlines()
        for i,name in enumerate(names):
            case_result_path=os.path.join(result_path,name)
            y2_path=os.path.join(case_result_path,'tooth2')
            y3_path=os.path.join(case_result_path,'tooth3')
            if not os.path.exists(case_result_path):
                os.mkdir(case_result_path)
                os.mkdir(y2_path)
                os.mkdir(y3_path)
            with open(y2_path+'/y.txt','w')as yf2:
                yf2.write(convert_line(output_list[2*i]))
            with open(y3_path+'/y.txt','w')as yf3:
                yf3.write(convert_line(output_list[2*i+1]))



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_path = '../ckpt'
    dir_save='/20190525-0016/rutine'
    save_checkpoints_dir = ckpt_path + '/' + dir_save

    var_file = save_checkpoints_dir+'/model.ckpt-7000'
    saver.restore(sess, var_file)  # 从模型中恢复最新变量
    state = [False, '']
    # data_fen = Data_Gen('F:/ProjectData/mesh_feature/tooth/save_npz/back')
    data_fen = Data_Gen('F:/ProjectData/mesh_feature/tooth_test/tooth/save_npz/back')
    data, case_num = data_fen.load_pkg(state)
    output_list=[]
    
    def convert_line(arr):
        list(arr)
        b = ['%.4f'%(i) for i in list(arr)]
        new_str = '-1 0 0 0'
        for i, num in enumerate(b):
            if i % 3 == 0:
                a = ','+str(int(i/3))+' '
            else:
                a = ' '
        
            new_str += (a + num)
        return new_str


    for i in range(case_num):
        feed_dict = build_feed_dict(plc, data, i)
        time_start = time.time()
        
        output_array = sess.run(output, feed_dict=feed_dict)
        
        output_list.append(output_array[:18])
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        
    wirte_result('F:/ProjectData/mesh_feature/tooth_test/tooth','case.txt',output_list)
        
        
