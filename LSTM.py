import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from SPXY import spxy

# 读取数据
data = pd.read_csv(open(r'../197n/数据/N_log9_wt.csv', 'r', encoding='utf-8'))

# 提取自变量和因变量
X = data.drop(['OM'], axis=1).values
Y = data['OM'].values

# 数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
minmax_x = min_max_scaler.fit_transform(X)

# 划分数据集
X_train, X_valid, y_train, y_valid = spxy(minmax_x, Y)

# 将数据转换为 3D 形状 (样本数, 时间步数, 特征数)
# 对于时间步数，我们这里设置为 1
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# 设置学习率
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# 训练LSTM模型
history = model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

# 在训练集上进行预测
y_pred_train = model.predict(X_train)

# 在验证集上进行预测
y_pred_valid = model.predict(X_valid)

# 计算R²
r2_train = r2_score(y_train, y_pred_train)
r2_valid = r2_score(y_valid, y_pred_valid)

# 计算RMSE
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

# 计算RPD
RPD_train = np.std(y_train) / RMSE_train  # 添加训练集的RPD计算
RPD_valid = np.std(y_valid) / RMSE_valid

print("训练集 R²:", r2_train)
print("验证集 R²:", r2_valid)
print("训练集 RMSE:", RMSE_train)
print("验证集 RMSE:", RMSE_valid)
print("训练集 RPD:", RPD_train)
print("验证集 RPD:", RPD_valid)

# 画散点图和1:1线
plt.figure(figsize=(8, 8))
plt.scatter(y_valid, y_pred_valid, c='b', label='Estimated value vs Measured value')
plt.plot([0, max(y_valid)], [0, max(y_valid)], 'r--', label='1:1 Line')
plt.text(0.1 * max(y_valid), max(y_valid) * 0.85, f'R²: {r2_valid:.3f}', fontsize=12)
plt.text(0.1 * max(y_valid), max(y_valid) * 0.8, f'RMSE: {RMSE_valid:.3f} g/kg', fontsize=12)
plt.text(0.1 * max(y_valid), max(y_valid) * 0.75, f'RPD: {RPD_valid:.3f}', fontsize=12)
plt.xlabel('Estimated value (g/kg)')
plt.ylabel('Measured value (g/kg)')
plt.legend()

plt.xlim(0, max(y_valid) * 1.05)
plt.ylim(0, max(y_valid))
plt.show()
