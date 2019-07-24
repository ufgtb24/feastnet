import numpy as np

from common.coarsening import adj_to_A, A_to_adj, coarsen, multi_coarsen


def read_adj(path,need_no):
    if need_no:
        adj=np.loadtxt(path).astype(np.int)[:,1:]
    else:
        adj = np.loadtxt(path).astype(np.int)
    return adj

def count_pairs(path):
    adj=read_adj(path,need_no=True)
    sum=np.sum((adj>0).astype(np.int))
    print(sum)
    
def get_adj(path):
    with open(path) as f:
        GG = f.readlines()
        N=len(GG)
        adj=np.zeros([N,14],dtype=np.int)
        for i,line in enumerate(GG):
            if line!='\n':
                for j,pt in enumerate(list(map(int, line.split()))):
                    adj[i,j]=pt
    return adj
    
def test_coarsen(adj_path):
    # adj=read_adj("F:/ProjectData/mesh_feature/adj.txt",need_no=False)
    adj=read_adj(adj_path,need_no=True)
    A = adj_to_A(adj)
    A.eliminate_zeros()  # 稀疏
    perm_in, A_out = coarsen(A, levels=2)
    print(len(perm_in))


    
def test_convert():
    adj=np.zeros([4,4])
    non_zero_idx=[[2,3,4],[1,3],[1,2,4],[1,3]]
    for i in range(4):
        for j,k in enumerate(non_zero_idx[i]):
            adj[i,j]=k
    print(adj.astype(np.int32))
    
    A=adj_to_A(adj)
    adj2=A_to_adj(4,A)
    print(adj2)


if __name__=="__main__":
    # adj_path="F:/ProjectData/mesh_direction/2aitest/low/BaileyKuo 5/adj.txt"
    adj_path="E:/VS_Projects/Mesh_Process2019/Test/adj.txt"
    
    # test_coarsen(adj_path)
    multi_coarsen(adj_path, 20, 2, 2)
    # count_pairs(adj_path)
    # adj=get_adj("F:/ProjectData/mesh_feature/test.txt")
    # adj=read_adj("F:/ProjectData/mesh_direction/2aitest/low/CassidyFraser 8/adj.txt",need_no=False)
    # A=adj_to_A(adj)
