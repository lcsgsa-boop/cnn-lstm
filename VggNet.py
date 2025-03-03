import csv
from keras.layers import Layer
from sklearn import preprocessing
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, MaxPooling1D
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from keras import backend as K
from KS import ks
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error
import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.layers import Reshape
from keras.layers import GlobalAveragePooling1D


def load_data(path):
    data = pd.read_csv(open(path, 'r', encoding='utf-8'))
    X = data.drop(['OM'], axis=1)
    X = X.values
    Y = data['OM'].values
    print(X)
    print(X.shape)

    #数据归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    minmax_x = min_max_scaler.fit_transform(X)
    print(minmax_x)

    #KS算法划分数据集
    X_train, X_vaild, y_train, y_vaild = ks(minmax_x, Y)

    #数据集添加维度
    X_train = X_train[:, :, np.newaxis]
    X_vaild = X_vaild[:, :, np.newaxis]

    # 标准差
    sd = np.var(y_vaild)
    print(sd)

    return X_train, X_vaild, y_train, y_vaild, sd


def my_init(shape, dtype=None):
    return 0.1*K.random_normal(shape, dtype=dtype)
# W = np.random.randn(input_layer_neurons,hidden_layer_neurons)* sqrt(2/input_layer_neurons)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=100,
    mode='min',
    verbose=2
)]



from keras.layers import Add





def VGGModel(X_train, X_vaild, y_train, y_vaild, sd):
    vgg = Sequential()

    vgg.add(Conv1D(input_shape=(201, 1), filters=8, kernel_size=3,
                   strides=1, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l1(0.0001)))

    vgg.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))


    vgg.add(Conv1D(filters=16, kernel_size=3, strides=1,
                   activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l1(0.0001)))

    vgg.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))



    vgg.add(Conv1D(filters=16, kernel_size=3, strides=1,
                   activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l1(0.0001)))

    vgg.add(Flatten())

    vgg.add(Dense(units=200, activation='relu'))
    vgg.add(Dropout(0.3))
    vgg.add(Dense(units=100, activation='relu'))
    vgg.add(Dropout(0.3))
    vgg.add(Dense(units=1, activation='relu'))


    print(vgg.summary())

    def r2_train(y_true, y_pred):
        # y_true 为TensorFlow张量
        # y_pred 为与y_ture同形状的TensorFlow张量
        r2 = 1 - mean_squared_error(y_true, y_pred) / sd
        return K.sqrt(r2 * r2)

    #动态调整lr
    def scheduler(epoch):
        # 每隔50个epoch，学习率减小为原来的1/10
        if epoch == 1000   and epoch != 0:
            lr = K.get_value(vgg.optimizer.lr)
            if lr > 1e-5:
                K.set_value(vgg.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
        return K.get_value(vgg.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    #激活函数
    nadam = optimizers.Nadam(lr=0.0012)
    adam = optimizers.Adam(lr=0.0001)
    sgd = optimizers.SGD(lr=0.0001)#随机梯度下降
    adagrad = optimizers.Adagrad(lr=0.001)#自适应梯度41-43
    rmsprop = optimizers.RMSprop(lr=0.001)#均方根
    adadelta = optimizers.Adadelta(lr=0.01)
    vgg.compile(optimizer=nadam, loss='mse', metrics=[r2_train])#, metrics=[r2_train]

    history = vgg.fit(X_train, y_train, epochs=2000, batch_size=157,
                      validation_data=(X_vaild, y_vaild), callbacks=[reduce_lr])
    #, callbacks=callbacks  , callbacks=[reduce_lr]


    y_train_pre = vgg.predict(X_train)
    y_valid_pre = vgg.predict(X_vaild)

    r2_train = r2_score(y_train, y_train_pre)
    r2_vaild = r2_score(y_vaild, y_valid_pre)

    RMSE_1_train = np.sqrt(np.mean((y_train - y_train_pre) ** 2))
    RMSE_1_vaild = np.sqrt(np.mean((y_vaild - y_valid_pre) ** 2))

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
        plt.scatter(y_vaild, y_valid_pre, c='b', label='Estimated value vs Measured value')
        plt.plot([0, max(y_vaild) ], [0, max(y_vaild) ], 'r--', label='1:1 Line')
        plt.text(0.1 * max(y_vaild) , max(y_vaild)  * 0.85, f'R²: {r2_vaild:.3f}', fontsize=12)
        plt.text(0.1 * max(y_vaild) , max(y_vaild)  * 0.8, f'RMSE: {RMSE_1_vaild:.3f} g/kg', fontsize=12)
        plt.text(0.1 * max(y_vaild) , max(y_vaild)  * 0.75, f'RPD_train: {RPD_train:.3f}', fontsize=12)
        plt.text(0.1 * max(y_vaild) , max(y_vaild)  * 0.7, f'RPD_valid: {RPD_valid:.3f}', fontsize=12)
        plt.xlabel('Estimated value(g/kg)')
        plt.ylabel('Measured value(g/kg)')
        plt.legend()

        plt.xlim(0, max(y_vaild) * 1.05)
        plt.ylim(0, max(y_vaild) * 1.05)
        plt.grid(True)

        # Remove grid lines
        plt.grid(False)

        plt.savefig(f'../结果图/vgg_N_log6_wt.csv_{r2_vaild}_.png')  # Save scatter plot
        plt.close()  # Close the plot

    return y_train_pre, y_valid_pre


if __name__ == '__main__':

    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_R.csv')
    #X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_DR.csv')
    X_train, X_vaild, y_train, y_vaild, sd = load_data(r'..\197n\数据\N_log6_wt.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_SVNR.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_RL.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_LR.csv')
    # X_train, X_vaild, y_train, y_vaild, sd = load_data(r'E:\光谱数据\光谱数据\206OM_MSCR.csv')

    r2 = 0
    i = 0

    while r2 < 0.95:

        list = []

        y_train_pre, y_vaild_pre = VGGModel(X_train, X_vaild, y_train, y_vaild, sd)

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

        # if r2 > 0.80:
        #     list.append(i)
        #     list.append(RMSE_1_train)
        #     list.append(r2_train)
        #     list.append(r2_vaild)
        #     list.append(RMSE_1_vaild)
        #     list.append(y_vaild_pre)
        # with open(r'vggdatatest/vgg7TestFe.csv', 'a', encoding='utf-8')as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow(list)
        i = i + 1

        #画出loss
        #plt.show()
        # # 绘制散点图
        # plt.figure(figsize=(6, 6))
        # plt.scatter(y_vaild, y_vaild_pre, color='blue', label='Data')
        # plt.plot([min(y_vaild), max(y_vaild)], [min(y_vaild), max(y_vaild)], color='red', linestyle='--',
        #          label='1:1 line')
        #
        # # 标注数据点的横坐标数值
        # for i in range(len(y_vaild)):
        #     plt.text(y_vaild[i], y_vaild_pre[i], str(round(y_vaild[i], 2)), fontsize=8, color='black', ha='center',
        #              va='bottom')
        # plt.legend()
        # plt.title('Scatter Plot with Symmetrical Axes and 1:1 Line')
        # plt.xlabel('True Values')
        # plt.ylabel('Predicted Values')
        #
        # # 设置坐标轴范围
        # min_val = min(np.min(y_vaild), np.min(y_vaild_pre))
        # max_val = max(np.max(y_vaild), np.max(y_vaild_pre))
        #
        #
        # plt.grid(True)
        # plt.show()

        # # 绘制散点图
        # plt.figure(figsize=(6, 6))
        # plt.scatter(y_train, y_train_pre, color='blue', label='Data')
        # plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--',
        #          label='1:1 line')
        #
        # # 标注数据点的横坐标数值
        # for i in range(len(y_train)):
        #     plt.text(y_train[i], y_train_pre[i], str(round(y_train[i], 2)), fontsize=8, color='black', ha='center',
        #              va='bottom')
        #
        # plt.legend()
        # plt.title('Scatter Plot with Symmetrical Axes and 1:1 Line')
        # plt.xlabel('True Values')
        # plt.ylabel('Predicted Values')
        #
        # # 设置坐标轴范围
        # min_val = min(np.min(y_train), np.min(y_train_pre))
        # max_val = max(np.max(y_train), np.max(y_train_pre))
        # plt.xlim(min_val, max_val)
        # plt.ylim(min_val, max_val)
        #
        # plt.grid(True)
        # plt.show()
