import os

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion

from feature_detect.src.config import *
from common.coarsening import multi_coarsen


def get_local(world_coord, center_coord, postfix):
    local_coord = world_coord - center_coord
    if postfix == 'DR':
        local_coord = local_coord * [-1, 1, 1]
    elif postfix == 'UL':
        local_coord = local_coord * [1, -1, 1]
    elif postfix == 'UR':
        local_coord = local_coord * [-1, -1, 1]
    return local_coord

def parse_vertice(vertice_file,center,postfix):
    v=np.loadtxt(vertice_file)
    v=get_local(v,center,postfix)
    return v
    
def parse_feature(feature_file, feat_cap, postfix):
    feat_arr = np.zeros([feat_cap, 4])
    
    with open(feature_file)as f:
        line = f.readline()
        feat_list = line.split(',')
        origin = None
        for feat3d in feat_list:
            feat3d_array = np.array(list(map(float, feat3d.split())))
            # feat3d_array=np.array(feat3d.split())
            feat_id = int(feat3d_array[0])  # 0 to feat_cap
            feat_coord = feat3d_array[1:]
            if feat_id == -1:
                origin = feat_coord
            else:
                feat_arr[feat_id][0] = 1
                feat_arr[feat_id][1:] = get_local(feat_coord ,origin,postfix)
    return feat_arr,origin


def save_np_data(data_path, idx_file,save_path, tasks, feat_cap,need_shufle=False):
    """
    processes the data into standard shape
    :param data_path: path_to_image box1,box2,...,boxN with boxX: x_min,y_min,x_max,y_max,class_index
    :param save_path: saver at "/home/minh/stage/train.npz"
    :param input_shape: (416, 416)
    :param max_boxes: 100: maximum number objects of an image
    :param load_previous: for 2nd, 3th, .. using
    :return: image_data [N, 416, 416, 3] not yet normalized, N: number of image
             box_data: box format: [N, 100, 6], 100: maximum number of an image
                                                6: top_left{x_min,y_min},bottom_right{x_max,y_max},class_index (no space)
                                                /home/minh/keras-yolo3/VOCdevkit/VOC2007/JPEGImages/000012.jpg 156,97,351,270,6
    """
    # return data['image_data'], data['box_data'], data['image_shape'], [data['y_true']]
    task_x = {task_name: [] for task_name in tasks.keys()}
    task_adj = {task_name: [] for task_name in tasks.keys()}
    task_perm = {task_name: [] for task_name in tasks.keys()}
    task_y = {task_name: [] for task_name in tasks.keys()}
    idx_file = os.path.join(data_path, idx_file)
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
            
            for postfix in os.listdir(filepath):
                part_path=os.path.join(filepath,postfix)
                print('part_path: ', part_path)

                for task_name, tooth_ids in tasks.items():
                    for tooth_id in tooth_ids:
                        print('tooth_id: ', tooth_id)
                        tooth_path = os.path.join(part_path, 'tooth%d' % (tooth_id))
                        if os.path.exists(tooth_path):
                            # [feat_cap,4]
                            feat_arr,center = parse_feature(os.path.join(tooth_path, 'feature.txt'), feat_cap=feat_cap,postfix=postfix)
                            task_y[task_name].append(feat_arr)
                            # [pt_num,3]
                            task_x[task_name].append(
                                parse_vertice(os.path.join(tooth_path, 'vertice.txt'),center,postfix))
                            
                            perms, adjs = multi_coarsen(os.path.join(tooth_path, 'adj.txt'), BLOCK_NUM, C_LEVEL)
                            task_adj[task_name].append(adjs)  # [pt_num,K]
                            task_perm[task_name].append(perms)  # [pt_num,K]
    
    for task_name in tasks.keys():
        task_path = os.path.join(save_path, task_name)
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        
        np.savez(task_path + '/data.npz',
                 x=np.array(task_x[task_name]),
                 adjs=np.array(task_adj[task_name]),
                 perms=np.array(task_perm[task_name]),
                 y=np.array(task_y[task_name])
                 )
def rotate(vertices, features, num_samples, rot_range, angle_fixed):
    '''
    
    :param vertices: [num_vertices, 3]
    :param features: [FEAT_CAP, 4]
    :param num_samples:
    :return:
    '''
    vertices=vertices.astype(np.float32)
    # [FEAT_CAP]
    mask=features[:,0].astype(np.bool)
    # [FEAT_CAP, 3]
    features=features[:,1:].astype(np.float32)
    if angle_fixed:
        vertices=vertices[np.newaxis, :, :]
        features=features[np.newaxis, :, :]
        return vertices, features, mask

    random_angles_x = np.random.uniform(-rot_range[0], rot_range[0],
                                      (num_samples)).astype(np.float32)
    random_angles_y = np.random.uniform(-rot_range[1], rot_range[1],
                                      (num_samples)).astype(np.float32)
    random_angles_z = np.random.uniform(-rot_range[2], rot_range[2],
                                      (num_samples)).astype(np.float32)

    # random_angles.shape: (num_samples, 3)
    random_angles=np.stack([random_angles_x,random_angles_y,random_angles_z],axis=1)
    ## debug
    regular_angles=np.concatenate([np.linspace(-np.pi, np.pi,num_samples)[:,np.newaxis],
                                   np.zeros((num_samples,2))],axis=-1).astype(np.float32)
    
    # random_quaternion.shape: (num_samples, 4)
    random_quaternion = quaternion.from_euler(random_angles)
    
    
    # vertices.shape : (num_samples, num_vertices, 3)
    vertices = quaternion.rotate(vertices[tf.newaxis, :, :],
                             random_quaternion[:, tf.newaxis, :]
                             )
    # features.shape : (num_samples, FEAT_CAP, 3)
    features = quaternion.rotate(features[tf.newaxis, :, :],
                             random_quaternion[:, tf.newaxis, :]
                             )
    
    return np.array(vertices),np.array(features),mask


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
    def __init__(self, rot_num,rot_range,angle_fixed, data_gen):
        self.rot_num = rot_num
        self.data_gen = data_gen
        self.rot_range=rot_range
        self.angle_fixed=angle_fixed
        self.ref_idx = -1
        self.rot_idx = -1
        self.load_data()
        # self.rotate_case()
    
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
        self.rot_vert, self.rot_feat,self.mask = rotate(
            self.data['x'][self.ref_idx],self.data['y'][self.ref_idx],
            self.rot_num,
            self.rot_range,
            self.angle_fixed
        )
        input_dict = {
            'vertice': self.rot_vert, #[rot_num,pt_num,3]
            'label': self.rot_feat, #[rot_num, FEAT_CAP, 3]
            'mask': self.mask, #[FEAT_CAP]
            'adjs':[self.data['adjs'][self.ref_idx][idx].astype(np.int32) for idx in range(self.block_num)],
            'perms':[self.data['perms'][self.ref_idx][idx].astype(np.int32) for idx in range(self.block_num-1)]
        }

        return input_dict,epoch_end
    

def get_idx(file_path):
    # 生成索引文件
    with open(os.path.join(file_path,'case.txt'),'w')as f:
        for file_name in os.listdir(file_path):
            if os.path.isdir(os.path.join(file_path,file_name)):
                f.write(file_name+'\n')
if __name__=='__main__':
    data_path='D:/py_Project/data/test'
    save_path='D:/py_Project/data/test_npz'
    # data_path='F:/ProjectData/mesh_feature/Case'
    # save_path='F:/ProjectData/mesh_feature/Case_npz'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # get_idx(data_path)
    save_np_data(data_path,'case.txt',save_path,TASKS,FEAT_CAP)
