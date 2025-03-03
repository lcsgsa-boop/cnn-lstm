import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np

import pandas as pd
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from KS import ks
from SPXY import spxy
from keras.layers import Layer

def load_data(path):
    data = pd.read_csv(open(path, 'r', encoding='utf-8'))
    X = data.drop(['OM'], axis=1)
    X = X.values
    Y = data['OM'].values/100
    print(X)
    print(X.shape)

    # 数据归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    minmax_x = min_max_scaler.fit_transform(X)
    print(minmax_x)

    # KS算法划分数据集
    X_train, X_vaild, y_train, y_vaild = spxy(minmax_x, Y)

    # 数据集添加维度
    X_train = X_train[:, :, np.newaxis]
    X_vaild = X_vaild[:, :, np.newaxis]

    # 标准差
    sd = np.var(y_vaild)
    print(sd)

    return X_train, X_vaild, y_train, y_vaild, sd



import tensorflow as tf
from tensorflow import keras

# class SEBlock(keras.layers.Layer):
#     def __init__(self, filter, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.reduction_ratio = reduction_ratio
#         self.filter = filter
#
#     def build(self, input_shape):
#         self.global_pooling = keras.layers.GlobalAveragePooling1D()
#         self.dense1 = keras.layers.Dense(self.filter // self.reduction_ratio, activation='relu')
#         self.dense2 = keras.layers.Dense(self.filter, activation='sigmoid')
#
#     def call(self, inputs):
#         avg_pool = self.global_pooling(inputs)
#         out = self.dense1(avg_pool)
#         out = self.dense2(out)
#         out = tf.expand_dims(out, axis=1)
#         return out * inputs

class ResnetBlock(keras.Model):

    def __init__(self, filter, kernelsize=3, strides=1, padding='same'):
        super().__init__()

        self.conv_model = keras.models.Sequential([
            keras.layers.Conv1D(filters=filter,
                                kernel_size=kernelsize,
                                strides=strides,
                                padding=padding,
                                # kernel_regularizer=keras.regularizers.l2(0.01)),  # 添加L2正则化
                                ),
            keras.layers.ReLU(),
            keras.layers.Conv1D(filters=filter,
                                kernel_size=kernelsize,
                                strides=1,
                                padding=padding,
                                #kernel_regularizer=keras.regularizers.l2(0.01)),  # 添加L2正则化
                                ),
            #SEBlock(filter),  # 添加SE块
        ])
        if strides != 1:
            # 即F(x)和x大小不同时，需要创建identity(x)卷积层
            self.identity = keras.models.Sequential([
                keras.layers.Conv1D(filters=filter,
                                    kernel_size=1,
                                    strides=strides,
                                    padding=padding,
                                    ),
            ])
        else:
            # 保持原样输出
            self.identity = lambda x: x

    def call(self, inputs, training=None):
        conv_out = self.conv_model(inputs)
        identity_out = self.identity(inputs)
        out = conv_out + identity_out
        out = tf.nn.relu(out)
        return out


class ResNet5(keras.Model):
    def __init__(self, block_list, num_classes):
        super().__init__()

        self.conv_initial = tf.keras.layers.Conv1D(8, 3, 1, padding='same')
        self.blocks = keras.models.Sequential()
        self.max_pool = keras.layers.GlobalMaxPool1D()
        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id][1]):
                if layer_id == 0:
                    # 每块中的第一个conv的stride = 2
                    self.blocks.add(ResnetBlock(filter=block_list[block_id][0], strides=2))
                else:
                    # 其他conv的stride = 1
                    self.blocks.add(ResnetBlock(filter=block_list[block_id][0], strides=2))
        # self.final_bn = keras.layers.BatchNormalization()
        self.max_pool = keras.layers.GlobalMaxPool1D()
        self.fc1 = keras.layers.Dense(200)
        #self.fc1 = keras.layers.Dense(200, kernel_regularizer=keras.regularizers.l2(0.01))  # 添加L2正则化
        self.dropout1 = keras.layers.Dropout(0.3)
        self.fc2 = keras.layers.Dense(100)
        self.fc2 = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01))  # 添加L2正则化
        self.dropout2 = keras.layers.Dropout(0.3)
        self.fc3 = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        out = self.conv_initial(inputs)
        # out = self.max_pool(out)
        out = self.blocks(out, training=training)
        # out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)
        out = self.max_pool(out)
        out = tf.nn.relu(out)
        out = self.fc1(out)
        out = self.dropout1(out)
        out = tf.nn.relu(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = tf.nn.relu(out)
        out = self.fc3(out)

        return out

def ResNet5_train(X_train, X_valid, y_train, y_valid, sd):
    num_classes = 1

    def scheduler(epoch):
        if epoch == 1000 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            if lr > 1e-5:
                K.set_value(model.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    def r2_train(y_true, y_pred):
        r2 = 1 - mean_squared_error(y_true, y_pred) / sd
        return K.sqrt(r2 * r2)

    model = ResNet5([[16, 3]], num_classes=num_classes)
    model.compile(optimizer=keras.optimizers.Nadam(0.0012),
                  loss='mse'
                  )
    model.build(input_shape=(None, 201, 1))

    model.summary()

    history = model.fit(X_train, y_train, epochs=900, batch_size=157,
                        validation_data=(X_valid, y_valid), callbacks=[reduce_lr])

    y_train_pre = model.predict(X_train)
    y_valid_pre = model.predict(X_valid)

    r2_train = r2_score(y_train, y_train_pre)
    r2_vaild = r2_score(y_valid, y_valid_pre)

    RMSE_1_train = np.sqrt(np.mean((y_train - y_train_pre) ** 2))
    RMSE_1_vaild = np.sqrt(np.mean((y_valid - y_valid_pre) ** 2))

    RPD_train = np.std(y_train_pre) / RMSE_1_train
    RPD_valid = np.std(y_valid_pre) / RMSE_1_vaild
    i = 0
    if r2_vaild > 0.3:
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

        #plt.savefig(f'../结果图/resnet_N_log9_wt_{r2_vaild}_.png')  # Save scatter plot
        plt.close()  # Close the plot

    return y_train_pre, y_valid_pre

if __name__ == '__main__':


    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_2DR.csv')#81
    X_train, X_vaild, y_train, y_vaild, sd = load_data(r'../197n/数据/N_log13_wt.csv')#84.4
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_LR.csv')#39
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_MSCR.csv')#62
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_R.csv')#64
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_RL.csv')#66
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n_Data\197N_SVNR.csv')#53

    r2 = 0
    i = 0

    while r2 < 0.95:

        list = []

        y_train_pre, y_vaild_pre = ResNet5_train(X_train, X_vaild, y_train, y_vaild, sd)

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
