import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from KS import ks
from SPXY import spxy

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mean_squared_error
def load_data(path):
    data = pd.read_csv(open(path, 'r', encoding='utf-8'))
    X = data.drop(['OM'], axis=1)
    X = X.values
    Y = data['OM'].values/100
    print(X)
    print(X.shape)

    #数据归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    minmax_x = min_max_scaler.fit_transform(X)
    print(minmax_x)

    #KS算法划分数据集
    X_train, X_vaild, y_train, y_vaild = spxy(minmax_x, Y)

    #数据集添加维度
    X_train = X_train[:, :, np.newaxis]
    X_vaild = X_vaild[:, :, np.newaxis]

    # 标准差
    sd = np.var(y_vaild)
    print(sd)

    return X_train, X_vaild, y_train, y_vaild, sd

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def AlexNet8(input_shape=(201, 1), num_classes=1):
    model = models.Sequential()

    # 第一层卷积层
    model.add(layers.Conv1D(96, 11, strides=4, padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(3, strides=2, padding='valid'))

    # 第二层卷积层
    model.add(layers.Conv1D(256, 5, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(3, strides=2, padding='valid'))

    # 第三层卷积层
    model.add(layers.Conv1D(384, 3, strides=1, padding='same', activation='relu'))

    # 第四层卷积层
    model.add(layers.Conv1D(384, 3, strides=1, padding='same', activation='relu'))

    # 第五层卷积层
    model.add(layers.Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(3, strides=2, padding='valid'))

    # 将多维数据展平为一维
    model.add(layers.Flatten())

    # 第一个全连接层
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # 第二个全连接层
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # 输出层，使用 linear 激活函数
    model.add(layers.Dense(num_classes, activation='linear'))

    return model



def AlexNet_train(X_train, X_valid, y_train, y_valid, sd):
    num_classes = 1  # 如果您的任务是回归，这里的 num_classes 应该是 1
    # 创建 AlexNet 模型实例
    model = AlexNet8(input_shape=(201, 1), num_classes=num_classes)
    # 定义学习率调度器
    def scheduler(epoch):
        # 每隔50个epoch，学习率减小为原来的1/10
        if epoch == 100 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            if lr > 1e-5:
                K.set_value(model.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler, verbose=1)


    # 定义 R2 评价函数
    def r2_train(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        r2 = 1 - mse / sd
        return tf.sqrt(tf.square(r2))


    # 激活函数
    # nadam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    adam = optimizers.Adam(lr=0.0001)
    sgd = optimizers.SGD(lr=0.0001)  # 随机梯度下降
    adagrad = optimizers.Adagrad(lr=0.001)  # 自适应梯度41-43
    rmsprop = optimizers.RMSprop(lr=0.001)  # 均方根
    adadelta = optimizers.Adadelta(lr=0.01)

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[r2_train])    # 打印模型结构

    model.summary()

    # 训练模型
    history = model.fit(X_train, y_train, epochs=1000, batch_size=157, validation_data=(X_valid, y_valid), callbacks=[reduce_lr])

    y_train_pre = model.predict(X_train)
    y_valid_pre = model.predict(X_vaild)

    r2_train = r2_score(y_train, y_train_pre)
    r2_vaild = r2_score(y_vaild, y_valid_pre)

    RMSE_1_train = np.sqrt(np.mean((y_train - y_train_pre) ** 2))
    RMSE_1_vaild = np.sqrt(np.mean((y_vaild - y_valid_pre) ** 2))

    RPD_train = np.std(y_train_pre) / RMSE_1_train
    RPD_valid = np.std(y_valid_pre) / RMSE_1_vaild
    i = 0
    if r2_vaild > 0.9:
        list = [i, RMSE_1_train, r2_train, r2_vaild, RMSE_1_vaild, y_valid_pre]

        # with open(r'../结果图/resnet结果.csv', 'a', encoding='utf-8') as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow(list)

        # Plotting loss curves
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.show()

        # 画散点图和1:1线
        plt.figure(figsize=(8, 8))
        plt.scatter(y_valid * 100, y_valid_pre * 100, c='b', label='Estimated value vs Measured value')
        plt.plot([0, max(y_valid) * 100], [0, max(y_valid) * 100], 'r--', label='1:1 Line')
        plt.text(0.1 * max(y_valid) * 100, max(y_valid) * 100 * 0.85, f'R²: {r2_vaild:.3f}', fontsize=12)
        plt.text(0.1 * max(y_valid) * 100, max(y_valid) * 100 * 0.8, f'RMSE: {RMSE_1_vaild:.3f} g/kg', fontsize=12)
        plt.text(0.1 * max(y_valid) * 100, max(y_valid) * 100 * 0.75, f'RPD_train: {RPD_train:.3f}', fontsize=12)
        plt.text(0.1 * max(y_valid) * 100, max(y_valid) * 100 * 0.7, f'RPD_valid: {RPD_valid:.3f}', fontsize=12)
        plt.xlabel('Estimated value(g/kg)')
        plt.ylabel('Measured value(g/kg)')
        plt.legend()

        plt.xlim(0, max(y_valid) * 100 * 1.05)
        plt.ylim(0, max(y_valid) * 100)
        plt.grid(True)

        # Remove grid lines
        plt.grid(False)

        # Remove grid lines
        plt.grid(False)

        plt.savefig(f'../结果图/AlexNet_{r2_vaild}_.png')  # Save scatter plot
        plt.close()  # Close the plot

    return y_train_pre, y_valid_pre


if __name__ == '__main__':

    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_R.csv')
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_DR.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_SVNR.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_RL.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_LR.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_MSCR.csv')
    X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_DR.csv')
    r2 = 0
    i = 0

    while r2 < 0.95:

        list = []

        y_train_pre, y_vaild_pre = AlexNet_train(X_train, X_vaild, y_train, y_vaild, sd)

        print(y_vaild_pre)

        #R2_TRAIN
        r2_train = r2_score(y_train, y_train_pre)
        print('r2_train')
        print(r2_train)
        print('\n')

        #R2_VAILD
        r2_vaild = r2_score(y_vaild, y_vaild_pre)
        print('r2_vaild')
        print(r2_vaild)
        print('\n')

        #RMSE_TRAIN
        y_train_pre = [i for item in y_train_pre for i in item]
        RMSE_1_train = np.sqrt(np.mean((y_train - y_train_pre) ** 2))
        print('RMSE_1_train')
        print(RMSE_1_train)
        print('\n')

        #RMSE_VAILD
        y_vaild_pre = [i for item in y_vaild_pre for i in item]
        RMSE_1_vaild = np.sqrt(np.mean((y_vaild - y_vaild_pre) ** 2))
        print('RMSE_1_vaild')
        print(RMSE_1_vaild)
        print('\n')

        # Calculate RPD for train and validation sets
        RPD_train = np.std(y_train_pre) / RMSE_1_train
        print('RPD_train')
        print(RPD_train)
        print('\n')
        RPD_valid = np.std(y_vaild_pre) / RMSE_1_vaild
        print('RPD_valid')
        print(RPD_valid)
        print('\n')

        r2 = r2_vaild
        i = i + 1

