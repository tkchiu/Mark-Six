import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载历史数据
data = np.loadtxt('data.csv', delimiter=',')

# 分割输入和输出
y1 = data[:, 1].values # 第一个开奖号码
y2 = data[:, 2].values # 第二个开奖号码
y3 = data[:, 3].values # 第三个开奖号码
y4 = data[:, 4].values # 第四个开奖号码
y5 = data[:, 5].values # 第五个开奖号码
y6 = data[:, 6].values # 第六个开奖号码
temperature = data[:, 7].reshape(-1, 1) # 温度特征
humidity = data[:, 8].reshape(-1, 1) # 湿度特征
sunrise = data[:, 9].reshape(-1, 1) # 日出时间特征
sunset = data[:, 10].reshape(-1, 1) # 日落时间特征
tide = data[:, 11].reshape(-1, 1) # 潮汐高度特征

# 对特征进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
temperature = scaler.fit_transform(temperature)
humidity = scaler.fit_transform(humidity)
sunrise = scaler.fit_transform(sunrise)
sunset = scaler.fit_transform(sunset)
tide = scaler.fit_transform(tide)

# 合并所有特征
X = np.hstack((X, temperature, humidity, sunrise, sunset, tide))

# 将输出转换为分类问题
y = np.zeros((data.shape[0], 49))
for i in range(data.shape[0]):
    for j in range(6):
        number = int(data[i, -6+j])
        y[i, number-1] = 1

# 构建神经网络模型
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
model.add(BatchNormalization()) # 添加BatchNormalization层,评估模型性能时，应该关闭
model.add(Dropout(0.2)) # 添加Dropout层,评估模型性能时，应该关闭
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization()) # 添加BatchNormalization层,评估模型性能时，应该关闭
model.add(Dropout(0.2)) # 添加Dropout层,评估模型性能时，应该关闭
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization()) # 添加BatchNormalization层,评估模型性能时，应该关闭
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=49, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X, y)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# 使用模型
new_data = np.array([[temperature, humidity, sunrise_time, sunset_time, tide_height]])
new_data = sc.transform(new_data)
prediction = model.predict(new_data)
print(prediction)
