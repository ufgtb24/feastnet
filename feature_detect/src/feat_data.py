import numpy as np
import os

from feature_detect.src.config import ADJ_K, BLOCK_NUM, TASKS, C_LEVEL, FEAT_CAP
from common.coarsening import adj_to_A, coarsen, A_to_adj, coarsen_index


def parse_feature(feature_file,feat_cap):
    
    
    feat_arr=np.zeros([feat_cap,4])
    
    with open(feature_file)as f:
        line=f.readline()
        feat_list = line.split(',')
        origin=None
        for feat3d in feat_list:
            feat3d_array=np.array(list(map(float, feat3d.split())))
            # feat3d_array=np.array(feat3d.split())
            feat_id = int(feat3d_array[0])  # 0 to feat_cap
            feat_coord=feat3d_array[1:]
            if feat_id==-1:
                origin=feat_coord
            else:
                feat_arr[feat_id][0]=1
                feat_arr[feat_id][1:]=feat_coord-origin
    return feat_arr



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
    task_x={task_name:[]for task_name in tasks.keys()}
    task_adj={task_name:[]for task_name in tasks.keys()}
    task_perm={task_name:[]for task_name in tasks.keys()}
    task_y={task_name:[]for task_name in tasks.keys()}
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

            filepath = os.path.join(data_path,case_name)
            if not os.path.exists(filepath):
                print("not found file " + filepath)
                continue
            for task_name,tooth_ids in  tasks.items():
                for tooth_id in tooth_ids:
                    print('tooth_id: ',tooth_id)
                    tooth_path=os.path.join(filepath,'tooth%d'%(tooth_id))
                    if os.path.exists(tooth_path):
                        task_x[task_name].append(
                            np.loadtxt(os.path.join(tooth_path,'x.txt'))[:,1:]) #[pt_num,3]
                        
                        perms, adjs=coarsen_index(os.path.join(tooth_path,'adj.txt'), ADJ_K, BLOCK_NUM, C_LEVEL)
                        task_adj[task_name].append(adjs) #[pt_num,K]
                        task_perm[task_name].append(perms) #[pt_num,K]
                        #[feat_cap,4]
                        feat_arr=parse_feature(os.path.join(tooth_path,'y.txt'),feat_cap=feat_cap)
                        task_y[task_name].append(feat_arr)

    for task_name in tasks.keys():
        task_path=os.path.join(save_path,task_name)
        if not os.path.exists(task_path):
            os.mkdir(task_path)
            
        np.savez(task_path + '/data.npz',
                 x=np.array(task_x[task_name]),
                 adj=np.array(task_adj[task_name]),
                 perm=np.array(task_perm[task_name]),
                 y=np.array(task_y[task_name])
                 )


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
        data = np.load(self.save_path + '/' + self.fileList[self.pkg_idx],allow_pickle=True)
        num=data['x'].shape[0]
        state[1] = self.fileList[self.pkg_idx] # 本次读取的 npz 文件名
        
        if self.pkg_idx < len(self.fileList) - 1:
            self.pkg_idx += 1
        else:
            self.pkg_idx = 0
            state[0] = True  #是否将所有所有数据读完，即一个epoch结束
        
        # return data['x'], data['adj'], data['perm'], data['y']
        return data,num


if __name__ == '__main__':
    data_path='F:/ProjectData/mesh_feature/tooth'
    save_path='F:/ProjectData/mesh_feature/tooth/save_npz'
    save_np_data(data_path,'case.txt',save_path,TASKS,FEAT_CAP)
    
    # data_fen=Data_Gen('F:/ProjectData/mesh_feature/tooth/save_npz/back')
    # state=[False,'']
    # while not state[0]:
    #     data=data_fen.load_pkg(state)
    #     print('')
    
