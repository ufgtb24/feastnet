CKPT_PATH= '../ckpt'


CHANNELS = [16,32, 64, 128]
TASKS={'back':[2,3],'mid':[4,5],'can':[6],'front':[7,8]}
# TASKS={'back':[2,3]}
FEAT_CAP=10  # 每个牙10个 feature 点
BLOCK_NUM= len(CHANNELS) - 1
C_LEVEL=3
# ADJ_K=20
