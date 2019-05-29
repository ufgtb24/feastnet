import os

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion

from Direction.src.config import *
from common.coarsening import multi_coarsen


def save_training_data(data_path,idx_file,save_dir,need_shufle=False):
    
    x_arr=[]
    adj_arr=[]
    perms_arr=[]
    idx_file=os.path.join(data_path,idx_file)

    with open(idx_file) as f:
        GG = f.read().splitlines()
        img_num = len(GG)
        if need_shufle:
            np.random.shuffle(GG)
    
        for case_name in (GG):
            if case_name == '\n':
                img_num -= 1
                continue
            if case_name.startswith('#'):
                img_num -= 1
                continue
            print('case_name: ', case_name)
        
            filepath = os.path.join(data_path, case_name)
            if not os.path.exists(filepath):
                print("not found file " + filepath)
                continue
            x_arr.append(np.loadtxt(os.path.join(filepath, 'x.txt')))  # [pt_num,3]
            perms, adjs = multi_coarsen(os.path.join(filepath, 'adj.txt'), ADJ_K, BLOCK_NUM, C_LEVEL)
            adj_arr.append(adjs)  # [pt_num,14]
            perms_arr.append(perms)
            
        save_path=os.path.join(data_path,save_dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.savez(save_path + '/data.npz',
                 x=np.array(x_arr),
                 adjs=np.array(adj_arr),
                 perms=np.array(perms_arr)
                 )


def generate_case_data(vertices, num_samples):
    # random_angles.shape: (num_samples, 3)
    random_angles = np.random.uniform(-np.pi, np.pi,
                                      (num_samples, 3)).astype(np.float32)
    
    # random_quaternion.shape: (num_samples, 4)
    random_quaternion = quaternion.from_euler(random_angles)
    
    # random_translation.shape: (num_samples, 3)
    random_translation = np.random.uniform(-2.0, 2.0,
                                           (num_samples, 3)).astype(np.float32)
    
    # data.shape : (num_samples, num_vertices, 3)
    data = quaternion.rotate(vertices[tf.newaxis, :, :],
                             random_quaternion[:, tf.newaxis, :]
                             ) + random_translation[:, tf.newaxis, :]
    
    # target.shape : (num_samples, 4+3)
    target = tf.concat((random_quaternion, random_translation), axis=-1)
    
    return np.array(data), np.array(target)


class Data_Gen():
    def __init__(self, save_path):
        files = os.listdir(save_path)
        self.save_path = save_path
        # npz文件名列表
        self.fileList = []
        self.pkg_idx = 0
        for f in files:
            if (os.path.isfile(save_path + '/' + f)):
                self.fileList.append(f)
    
    def load_pkg(self, state):
        print("load file: " + self.fileList[self.pkg_idx])
        data = np.load(self.save_path + '/' + self.fileList[self.pkg_idx], allow_pickle=True)
        num = data['x'].shape[0]
        state[1] = self.fileList[self.pkg_idx]  # 本次读取的 npz 文件名
        
        if self.pkg_idx < len(self.fileList) - 1:
            self.pkg_idx += 1
        else:
            self.pkg_idx = 0
            state[0] = True  # 是否将所有所有数据读完，即一个epoch结束
        
        # return data['x'], data['adj'], data['perm'], data['y']
        return data, num


if __name__=='__main__':
    data_path="F:/ProjectData/mesh_direction/2aitest/low"
    save_training_data(data_path,'case_list.txt','npz')
    # dg=Data_Gen()