import keras
from keras import applications
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import os


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




# 获取  VGG-19
def importVGG():
    # model_vgg19 = applications.VGG19(
    #     # 这里是利用预训练的模型来做特征提取
    #     # 因此我们不需要顶层的分类器网络部分的权重，只需要使用到训练好的卷积基。
    #     # 这也就是VGG19参数中include_top=False的含义
    #     include_top=False,
    #     weights='../models/vgg-19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # )
    # model_vgg19.trainable = False
    base_model = keras.Sequential()
    core_model = VGG19(weights='imagenet',include_top=True)
    print("\nVGG-19:\n")
    print(core_model.summary())
    base_model.add(core_model)
    output = keras.layers.Dense(512, activation='sigmoid')
    base_model.add(output)
    print("\nbase_model:\n")
    print(base_model.summary())
    return base_model




def extractFeature(imagesDir,model):
    print("Images' directory: " + imagesDir)
    list = []
    for num in range(1, 501):
        img_path = imagesDir + str(num) + '.jpg'
        print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))  # 加载图像，归一化大小
        x = image.img_to_array(img)  # 序列化
        x = np.expand_dims(x, axis=0)  # 展开
        x = preprocess_input(x)  # 预处理到0～1
        result = model.predict(x)  # 预测结果，512维的向量
        feature = np.array(result).tolist()
        list = list + feature
    min_max_scaler = preprocessing.MinMaxScaler()  # sklearn 归一化
    image_feature = pd.DataFrame(data=list)
    image_feature_normed = min_max_scaler.fit_transform(image_feature)
    image_feature_normed = pd.DataFrame(data=image_feature_normed.tolist())
    root_dir = os.path.dirname(os.path.abspath('.')) # 到项目根目录
    image_feature_normed.to_csv(root_dir+'\\advance\outputs\\feature\\imageFeature.csv')




def extract(imageDataPath):
    print("===============================extract===============================");
    VGG19 = importVGG()
    extractFeature(imageDataPath,VGG19)












