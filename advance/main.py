
import os
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import keras.backend.tensorflow_backend as KTF #进行配置，每个GPU使用80%上限现存
from tensorflow.python.client import device_lib



from advance import saveToCSV
from advance.image_feature_extraction import extract
from advance.music_feature_extraction import getMusicData
from advance.tests import test
from advance import train_old_model

def main():
    print("\nInprove of Image-music-synesthesia-aware\n");
    imageDataPath = "E:\learn\music_audio\emotionData\image\jpg\\"
    root_dir = os.path.dirname(os.path.abspath('.'))  # 到项目根目录

    # 提取图像特征  目前 512 x 1
    if (os.path.exists(root_dir+'\\advance\outputs\\feature\\imageFeature.csv') == False):
        extract.extract(imageDataPath)
    else:
        print("Image feature has been extracted!")

    # 提取音频特征
    # 由 Sound-Net 提取  ## 500 * 1203
    musicFeature = getMusicData.extract()
    X_scaler = StandardScaler()
    musicFeature = X_scaler.fit_transform(musicFeature)

    # 特征融合
    # TODO new way
    label_data = pd.read_csv(open("E:/learn/music_audio/emotionData/pearson.csv", encoding='utf-8')).values
    music_data = musicFeature
    image_data = pd.read_csv(open("./outputs/feature/imageFeature.csv", encoding='utf-8')).values
    print(label_data.shape)
    print(music_data.shape) # (500, 1203)
    #print(image_data.shape) # (500, 1204) 有序号
    xx = image_data
    xx = image_data.transpose()
    xx = np.delete(xx,0,axis=0)
    image_data = xx.transpose()
    print(image_data.shape) # (500, 1203) 无序号

    ## 存储到 CSV 文件
    if (os.path.exists(root_dir+'/advance/outputs/feature/image_music_label.csv')):
        print("特征已提取写入！")
    else:
        # TODO 耗时太多 分部分写入
        list_1 = saveToCSV.wirteFinalDataToCSV_1(label_data_frame=label_data, music_data_mat=music_data, image_data_mat=image_data)
        dataFrame = pd.DataFrame(list_1)
        print(dataFrame.shape)
        dataFrame.to_csv(root_dir+'/advance/outputs/feature/image_music_label.csv')


    # 训练预测模型
    # TODO

    train_old_model.train_old_model(root_dir+'/advance/outputs/feature/image_music_label.csv','E:/learn/music_audio/emotionData/pearson.CSV')


# 查看是否有GPU
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
# 打印
#     print(local_device_protos)

# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']

# GPU是否可用  返回True或者False
tf.test.is_gpu_available()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为0号的GPU



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 每个GPU现存上届控制在80%以内
session = tf.Session(config=config)  # 设置session KTF.set_session(session )
# 设置session
KTF.set_session(session)


main()
test.test()

