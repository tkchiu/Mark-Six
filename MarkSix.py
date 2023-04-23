import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 生成包含1-49的数字列表
numbers = list(range(1, 50))

# 将数字列表转换为one-hot编码的矩阵
def to_one_hot(numbers):
    n = len(numbers)
    one_hot = np.zeros((n, n))
    for i in range(n):
        one_hot[i, i] = 1
    return one_hot

numbers_one_hot = to_one_hot(numbers)

# 随机生成一些样本数据
X_train = np.random.choice(numbers_one_hot, size=(10000, 6), replace=True)
y_train = np.random.choice(numbers, size=(10000, 6), replace=True)

# 构建神经网络模型
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=49))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

# 编译模型并训练
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, to_one_hot(y_train), epochs=10, batch_size=32)
