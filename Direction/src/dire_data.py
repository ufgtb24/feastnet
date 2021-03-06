import os

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion

from Direction.src.config import *
from common.coarsening import multi_coarsen


def process_data(data_path, idx_file, save_dir=None, npz_name='data.npz', need_shufle=False):
    
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
            # last block don't need pooling, only conv,so coarsen times is block_num-1
            perms, adjs = multi_coarsen(os.path.join(filepath, 'adj.txt'), ADJ_K, BLOCK_NUM - 1, C_LEVEL)
            adj_arr.append(adjs)  # [5,pt_num,14]
            perms_arr.append(perms)
        
        x=np.array(x_arr)
        adjs = np.array(adj_arr)
        perms = np.array(perms_arr)

        if save_dir is not None:
            save_path=os.path.join(data_path,save_dir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
    
            np.savez(os.path.join(save_path,npz_name),
                     x=x,
                     adjs=adjs,
                     perms=perms
                     )
        else:
            return x,adjs,perms


def rotate(vertices, num_samples):
    vertices=vertices.astype(np.float32)
    # random_angles.shape: (num_samples, 3)
    random_angles = np.random.uniform(-np.pi, np.pi,
                                      (num_samples, 3)).astype(np.float32)
    
    ## debug
    regular_angles=np.concatenate([np.linspace(-np.pi, np.pi,num_samples)[:,np.newaxis],
                                   np.zeros((num_samples,2))],axis=-1).astype(np.float32)
    
    ##
    
    # random_quaternion.shape: (num_samples, 4)
    random_quaternion = quaternion.from_euler(random_angles)
    
    
    # data.shape : (num_samples, num_vertices, 3)
    data = quaternion.rotate(vertices[tf.newaxis, :, :],
                             random_quaternion[:, tf.newaxis, :]
                             )
        
    return np.array(data), np.array(random_quaternion)


class Data_Gen():
    def __init__(self, save_path):
        files = os.listdir(save_path)
        self.save_path = save_path
        # npz文件名列表
        self.fileList = []
        self.pkg_idx = -1
        for f in files:
            if (os.path.isfile(save_path + '/' + f)):
                self.fileList.append(f)
    
    def load_pkg(self):
        epoch_end=False
        if self.pkg_idx==len(self.fileList)-1:
            epoch_end=True
            self.pkg_idx=-1

        self.pkg_idx += 1
        npz_name=self.fileList[self.pkg_idx]
        data = np.load(self.save_path + '/' + npz_name, allow_pickle=True)
        # print("load file: " + npz_name)
        return data, npz_name ,epoch_end


class Rotate_feed():
    def __init__(self, rot_num, data_gen):
        self.rot_num = rot_num
        self.data_gen = data_gen
        self.ref_idx = -1
        self.rot_idx = -1
        self.load_data()
        self.rotate_case()
    
    def load_data(self):
        self.data,  npz_name,epoch_end = self.data_gen.load_pkg()
        self.case_num = self.data['x'].shape[0]
        self.block_num = self.data['adjs'].shape[1]
        return epoch_end
    
    def rotate_case(self):
        epoch_end=False
        if self.ref_idx == self.case_num-1:
            self.ref_idx = -1
            epoch_end=self.load_data()
        
        self.ref_idx += 1
        self.rot_vert, self.rot_quat = rotate(self.data['x'][self.ref_idx], self.rot_num)
        input_dict = {
            'vertice': self.rot_vert, #[rot_num,pt_num,3]
            'label': self.rot_quat, #[rot_num,4]
            'ori_vertice':self.data['x'][self.ref_idx].astype(np.float32),
            'adjs':[self.data['adjs'][self.ref_idx][idx].astype(np.int32) for idx in range(self.block_num)],
            'perms':[self.data['perms'][self.ref_idx][idx].astype(np.int32) for idx in range(self.block_num-1)]
        }

        return input_dict,epoch_end
    


if __name__=='__main__':
    # data_path="/home/yu/Documents/project_data/low"
    data_path="F:/ProjectData/mesh_direction/2aitest/low"
    # process_data(data_path, 'case_test.txt', 'npz_test', 'data.npz')
    process_data(data_path, 'case_list.txt', 'npz', 'data.npz')
