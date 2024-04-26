# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.fashion_mnist import load_fashionMNIST
from deepCNN import DeepCNN
import matplotlib.pyplot as plt
from common.optimizer import *
from common.trainer import Trainer

# for reproducibility
np.random.seed(1)

# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
(x_train, t_train), (x_test, t_test) = load_fashionMNIST(normalize=True, flatten=False, one_hot_label=True)

network = DeepCNN()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=30, mini_batch_size=350,
                  optimizer='Adam', optimizer_param={'lr':0.0014},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()


# 파라미터 저장
path_dir = './ckpt'
file_name = "deep_params.pkl"
if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

network.save_params(os.path.join(path_dir, file_name))
print("Parameter Save Complete!")