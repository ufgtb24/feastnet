import numpy as np
import os

from config import ADJ_K, LAYER_NUM, TASKS
from src.coarsening import adj_to_A, coarsen, A_to_adj



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

def coarsen_index(adj_path,coarsen_times,coarsen_level):
    '''
    
    :param adj_path:
    :param coarsen_times:
    :param coarsen_level:
    :return perms: [np.array([size_to_sample_from])]*coarsen_times
    :return adjs: [np.array([pt_num,ADJ_K])]*coarsen_times
    
    '''
    adj=np.loadtxt(adj_path).astype(np.int)[:,1:]
    perms=[]
    adjs=[]
    adjs.append(adj)
    for i in range(coarsen_times):
        print('c_time: ',i)
        A=adj_to_A(adj)
        perm,A=coarsen(A, coarsen_level)
        perms.append(perm)
        adj=A_to_adj(ADJ_K,A) # TODO 需要小 K ?
        adjs.append(adj)

    return np.array(perms),np.array(adjs)


def save_np_data(data_path, save_path, tasks, feat_cap,
                 pkg_capacity=2000, need_shufle=False):
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
    idx_file=os.path.join(data_path,'case.txt')
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
                        
                        perms, adjs=coarsen_index(os.path.join(tooth_path,'adj.txt'),LAYER_NUM,2)
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

def get_training_data(root_path, load_previous=True):
    load_path = root_path + '/train.npz'
    if load_previous == True and os.path.isfile(load_path):
        data = np.load(load_path)
        print('Loading training data from ' + load_path)
        return data['X'], data['Adj'], data['Pidx'], data['Pedge'], \
               data['Y'], data['Ynm'], data['Mask']
    
    sample_dir = os.listdir(root_path)
    x_list = []
    adj_list = []
    p_idx_list = []
    p_edge_list = []
    y_list = []
    y_nm_list = []
    mask_list = []
    
    for sample in sample_dir:
        sample_path = os.path.join(root_path, sample)
        if os.path.isdir(sample_path):
            x = np.loadtxt(os.path.join(sample_path, 'x.txt'))[:, 1:].astype(np.float32)
            adj = np.loadtxt(os.path.join(sample_path, 'x_ad.txt'))[:, 1:11].astype(np.int32)
            p_idx = np.loadtxt(os.path.join(sample_path, 'x_add_idx.txt')).astype(np.int32)
            
            p_adj = adj[p_idx - 1]
            p_edge = extract_edge(p_idx, p_adj)
            mask = build_mask(x.shape[0], p_idx)
            
            y_nm = np.loadtxt(os.path.join(sample_path, 'y_normal.txt')).astype(np.float32)
            y = y_nm[:, :3]
            y_nm = y_nm[:, 3:]
            
            x_list.append(x)
            adj_list.append(adj)
            p_idx_list.append(p_idx)
            p_edge_list.append(p_edge)
            y_list.append(y)
            y_nm_list.append(y_nm)
            mask_list.append(mask)
    
    X = np.array(x_list)
    Adj = np.array(adj_list)
    Pidx = np.array(p_idx_list)
    Pedge = np.array(p_edge_list)
    Y = np.array(y_list)
    Ynm = np.array(y_nm_list)
    Mask = np.array(mask_list)
    if not load_previous:
        np.savez(load_path, X=X, Adj=Adj, Pidx=Pidx, Pedge=Pedge,
                 Y=Y, Ynm=Ynm, Mask=Mask)
    
    return X, Adj, Pidx, Pedge, Y, Ynm, Mask

if __name__ == '__main__':
    data_path='F:/ProjectData/mesh_feature/tooth'
    save_path='F:/ProjectData/mesh_feature/tooth/save_npz'
    case_path=os.path.join(data_path,'Feature MO143Initial')
    print(case_path)
    print(os.path.exists(case_path))
    save_np_data(data_path,save_path,TASKS,10)
