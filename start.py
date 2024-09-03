import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------ 读取数据 ------------------------------ 
data = pd.read_excel('data.xlsx', parse_dates=['Date'], index_col='Date')

# ------------------------------ 数据预处理 ------------------------------
# 将 NAS100 列向上移动一行，创建 t+1 日的 NAS100 数据列
data['NAS100_t+1'] = data['NAS100'].shift(-1)
data = data.dropna()
model_data = data[['NAS100_t+1', 'DJI', 'Unexp', 'TB3M']]

# 划分训练集和验证集
training_data = model_data[(model_data.index >= '2014-01-02') & (model_data.index <= '2020-12-31')]
validation_data = model_data[(model_data.index >= '2021-01-01') & (model_data.index <= '2021-12-31')]

# 分别对特征和目标变量进行归一化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# 对特征进行归一化
scaled_X = scaler_X.fit_transform(model_data.iloc[:, 1:])

# 对目标变量进行归一化
scaled_y = scaler_y.fit_transform(model_data.iloc[:, 0:1])

# 结合特征和目标变量
scaled_data = np.concatenate([scaled_y, scaled_X], axis=1)

# 定义时间步长（time_step）
time_step = 5

# 创建 LSTM 输入格式的数据
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 1:])  # 从第二列开始（自变量）
        Y.append(data[i + time_step, 0])  # 第一列是目标变量
    return np.array(X), np.array(Y)

# 将数据转换为 LSTM 输入格式
X, Y = create_dataset(scaled_data, time_step)

# 使用前面的切片操作，生成训练集和验证集
train_size = len(training_data) - time_step
X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:], Y[train_size:]

# 重塑数据以符合 LSTM 输入要求：[样本数, 时间步长, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))

# ------------------------------ 构建 LSTM 模型 ------------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=2)

# 在验证集上进行预测
predicted = model.predict(X_val)

# 仅对预测的目标变量进行反归一化
predicted = scaler_y.inverse_transform(predicted)
Y_val = scaler_y.inverse_transform(Y_val.reshape(-1, 1))

# ------------------------------ 评估模型性能 ------------------------------
mse = mean_squared_error(Y_val, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_val, predicted)
r2 = r2_score(Y_val, predicted)

print("\n模型性能指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# ------------------------------ 可视化预测结果 ------------------------------
plt.figure(figsize=(14, 7))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保 x 和 y 轴数据匹配
# 获取验证集的日期索引
x_axis = validation_data.index[time_step:]  # 从 time_step 开始的日期，长度与 Y_val 匹配

# 确保 Y_val 对应的日期索引与 x_axis 匹配
Y_val = Y_val[:len(x_axis)]  # 确保 Y_val 的长度与 x_axis 匹配

# 对预测值也进行类似的处理
predicted = predicted[:len(x_axis)]

plt.plot(x_axis, Y_val, label='实际NAS100收盘价 (t+1)', color='blue')
plt.plot(x_axis, predicted, label='预测NAS100收盘价 (t+1)', color='red', linestyle='--')

plt.title('实际vs预测NAS100收盘价 (t+1)')
plt.xlabel('日期')
plt.ylabel('收盘价')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

ax.tick_params(axis='x', which='major', pad=15)
ax.tick_params(axis='x', which='minor', labelsize=8)

plt.legend()
plt.grid(True)
plt.show()
