import numpy as np
from numpy import *

# 只写入了图片和音频的特征
def processData(image_data_mat, music_data_mat, image_num, audio_num, pearson):
    list = []
    a = image_data_mat[image_num - 1].tolist()
    b = music_data_mat[audio_num - 1].tolist()
    list = a + b
    dataMat = mat(list)
    print(dataMat.shape)
    return list


def wirteFinalDataToCSV_1(label_data_frame,music_data_mat,image_data_mat):
    list = []
    flag = True
    for i in range(0, 250000):
        str_x = label_data_frame[i]
        print(i + 1)
        print("image: " + str_x[:1][0][1:] + "    audio: " + str_x[1:2][0][1:] + "   pearson: " + str(str_x[3:4][0]))
        if flag == True:
            list = np.mat(processData(image_data_mat, music_data_mat, int(str_x[:1][0][1:]), int(str_x[1:2][0][1:]), str_x[3:4][0]))
            flag = False
        else:
            list = np.vstack((list, processData(image_data_mat, music_data_mat, int(str_x[:1][0][1:]), int(str_x[1:2][0][1:]), str_x[3:4][0])))
        print()
    return list