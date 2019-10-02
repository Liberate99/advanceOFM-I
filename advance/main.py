import os
from image_feature_extraction import extract
from tests import test

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
    # 由 Sound-Net 提取


    # 特征融合
    #

    # 训练预测模型
    # TODO

main()
test.test()

