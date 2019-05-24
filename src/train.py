from __future__ import division

import os

import tensorflow as tf
import numpy as np
import math
import time
import h5py
import argparse

from config import FEAT_CAP, BLOCK_NUM, CHANNELS
from src.coarsening import adj_to_A, coarsen, A_to_adj
from src.data_process import get_training_data
from src.loss_func import loss_func
from src.model import *
from src.utils import *

random_seed = 0
np.random.seed(random_seed)

sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=int, default=0)
parser.add_argument('--dataset_path')
parser.add_argument('--results_path')
parser.add_argument('--num_iterations', type=int, default=50000)
parser.add_argument('--num_input_channels', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_points', type=int)
parser.add_argument('--num_classes', type=int)

FLAGS = parser.parse_args()

ARCHITECTURE = FLAGS.architecture
DATASET_PATH = FLAGS.dataset_path
RESULTS_PATH = FLAGS.results_path
NUM_ITERATIONS = FLAGS.num_iterations
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
LEARNING_RATE = FLAGS.learning_rate
BATCH_SIZE = 16
K = 4
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

"""
Load dataset 
x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
										  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
										  e.g. [16,10,4] 16 batch, 10 vertice with 4 neib for each
"""

def build_input(block_num):
    adjs=[]
    perms=[]
    input=tf.placeholder(tf.float32,[None,3])
    label=tf.placeholder(tf.float32,[FEAT_CAP,4])
    for i in range(block_num):
        adjs.append(tf.placeholder(tf.int32,[None,K]))
        perms.append(tf.placeholder(tf.int32,[None,K]))
    return {'input':input,'label':label,'perms':perms,'adjs':adjs}
        
        
plc=build_input(BLOCK_NUM)
output = get_model(plc,CHANNELS, FEAT_CAP )
loss=loss_func(output,plc['label'])
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Checkpoint restored\n")

# Train for the dataset


for iter in range(NUM_ITERATIONS):
    for i in range(case_num):
        A_0 = adj_to_A(adj_train[i])
        perm_0,A_1=coarsen(A_0, x_train[i], COARSEN_LEVEL)
        adj_1=A_to_adj(NUM_POINTS,K,A_1)
        
        
        _,loss_train = sess.run([train_step,cross_entropy], feed_dict={
            x: x_train[i],
            adj0: adj_train[i],
            adj1:adj_1,
            perm0:perm_0,
            y: y_train[i]})
        
    np.random.shuffle(perm)  # 打乱
    x_train = x_train[perm]
    adj_train = adj_train[perm]
    y_train = y_train[perm]

    if iter % 1000 == 0:
        A_0 = adj_to_A(adj_valid[0])
        perm_0,A_1=coarsen(A_0, x_valid[0], COARSEN_LEVEL)
        adj_1=A_to_adj(NUM_POINTS,K,A_1)

        loss_train = sess.run([cross_entropy], feed_dict={
            x: x_valid[0],
            adj0: adj_valid[0],
            adj1:adj_1,
            perm0:perm_0,
            y: y_valid[0]})
