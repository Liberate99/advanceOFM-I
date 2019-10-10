# -*- coding: utf-8 -*-
'''
@Time    : 2019/10/10 10:26
@Author  : DJ
@File    : train_old_model.py
'''
import pandas as pd
import tensorflow as tf
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger)
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K

from sklearn import metrics
from sklearn.model_selection import train_test_split
import csv
import keras

# create callbacks list
LR_FACTOR = 0.2
PATIENCE = 2
checkpoint_file = './working/resnet50_focal.h5'


callbacks_list = [
    # 训练可视化
    keras.callbacks.TensorBoard(
        log_dir='./logs',  # log 目录
        histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
        # batch_size=32,     # 用多大量的数据计算直方图
        write_graph=True,  # 是否存储网络结构图
        write_grads=True,  # 是否可视化梯度直方图
        write_images=True,  # 是否可视化参数
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None),

    # 训练自动更改学习率
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_mean_squared_error',
        factor=0.2,
        patience=5,
        verbose=1,
        mode='min',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.0001),

    # 当被监测的数量不再提升，则停止训练
    keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,
        patience=30,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=False)
]

def designModel():
    print("====================designModel=======================")
    model = Sequential()

    model.add(Dense(2406, input_dim=2406, activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1024, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(512, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(256, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(256, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(128, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(128, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(64, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(32, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(16, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(16, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(8, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(8, kernel_initializer='random_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            ))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error',
                 'mean_absolute_error',
                 'mean_absolute_percentage_error',
                 'mean_squared_logarithmic_error',
                 r_square_final,
                 # metrics.r2_score,
                 # metrics.r2_score,
                 pearson_r,
                 mre]
    )

    print(model.summary())

    return model


def mre(y_true, y_pred):
    return K.mean(K.sum(K.abs(100*K.ones_like(y_true) * (y_true - y_pred) / y_true)))


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def r_square_final(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    return SSR/SST

# 训练
def train_old_model(featurePath, pearsonPath):
    print("=================train_old_model==================")

    # path stracture:  image: 1203,

    X = pd.read_csv(open(featurePath, encoding='utf-8'),usecols = range(1,2407))
    X = X.values
    print("feature dimension:  ")
    print(X.shape)


    with open(pearsonPath, "rt", encoding="utf-8") as vsvfile: #'../../emotionData/pearson.CSV'
        reader = csv.reader(vsvfile)
        y = [row[3] for row in reader]
    # y = pd.DataFrame(y)
    y = y[1:]
    y = pd.DataFrame(y)
    print("label dimension:  ")
    print(y.shape)

    old_model = designModel()

    keras.backend.get_session().run(tf.global_variables_initializer())

    his = old_model.fit(
        X,
        y,

        validation_split=0.1,
        epochs=300,
        batch_size=1024,

        shuffle=True,
        verbose=1,

        callbacks=callbacks_list
    )

    # calculate predictions

    test_X = X
    # actual = y
    # model = load_model('model.h5')
    predictions = old_model.predict(test_X)

    # round predictions
    B = []
    for x in predictions:
        for i in x:
            B.append(i)

    B = B[1:250000]

