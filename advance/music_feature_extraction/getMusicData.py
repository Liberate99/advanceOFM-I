# -*- coding: utf-8 -*-
'''
@Time    : 2019/10/2 20:28
@Author  : DJ
@File    : extract.py
'''
import os
import numpy as np

## 读取提取好的特征

def extract():
    firldir = "E:\learn\music_audio\SoundNet-tensorflow\output"
    os.listdir(firldir)
    feature = np.load(firldir + "\1\tf_fea26_1")

    print(feature)
    print(feature.shape)