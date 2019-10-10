# -*- coding: utf-8 -*-
'''
@Time    : 2019/10/2 20:28
@Author  : DJ
@File    : getMusicData.py
'''
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


## 读取提取好的特征
def extract():
    firldir = "E:\learn\music_audio\SoundNet-tensorflow\output"
    os.listdir(firldir)
    list = []
    for i in range(1,6):
        for j in range(1,101):
            path = firldir + "\\" + str(i) +"\\tf_fea26_" + str(j) + ".npy"
            if os.path.exists(path):
                feature = np.load(path)
                # reshapeFeature(feature)
                # print(feature)
                print("file:  " + path)
                data = reshapeFeature(feature)
                list.append(data)
            else:
                print("file:  " + path + "  文件不存在")
                arr = np.zeros([1203,])
                list.append(arr)
    return list


## 降维：
'''
线性
    PCA
    LDA
非线性
    保留局部特征
    保留全局特征
'''


## reshape feature
def reshapeFeature(feature):
    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(feature)
    # 矩阵进行转置
    x_1 = x.transpose()
    # PCA
    pca = PCA(n_components=3)  # 保证降维后的数据保持90%的信息
    x_2 = pca.fit_transform(x_1)
    x_3 = x_2.flatten()
    print(x_3.shape)
    return x_3