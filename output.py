import numpy as np
from tensorflow.keras.models import load_model

# 加载已经训练好的神经网络模型
model = load_model('your_model_path.h5')

# 准备输入特征并进行预处理
input_data = np.array([[1, 2, 3, 4, 5, 6, 20, 30, 25, 7.5]])
scaler = StandardScaler()
input_data[:, :-4] = scaler.fit_transform(input_data[:, :-4])
input_data[:, -4:] = scaler.transform(input_data[:, -4:])

# 进行预测
output = model.predict(input_data)

# 从预测结果中选择概率最大的6个数字作为预测结果
result = []
for i in range(6):
    max_index = np.argmax(output)
    result.append(max_index+1)
    output[0, max_index] = 0

print('预测结果为：', result)
