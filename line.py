# keras 
# 1. 求解线性回归
# 2. r方检查测试结果是否符合直线（不能预测准确性）

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

# %matplotlib inline

TRUE_W = 3.0
TRUE_b = 2.0
NUM_SAMPLES = 100

# 初始化随机数据
X = tf.random.normal(shape=[NUM_SAMPLES, 1]).numpy()
noise = tf.random.normal(shape=[NUM_SAMPLES, 1]).numpy()
y = X * TRUE_W + TRUE_b + noise  # 添加噪声

model = tf.keras.Sequential()  # 新建顺序模型
model.add(tf.keras.layers.Dense(units=1, input_dim=1))  # 添加线性层
model.compile(optimizer='sgd', loss='mse')  # 指定损失函数为 MSE 平方损失函数，优化器选择 SGD 随机梯度下降
model.summary()  # 查看模型结构
# batch_size 经验法则调整 * 2
model.fit(X, y, epochs=500, batch_size=32)  # 训练模型

# 预测
x_test = tf.constant([1,2], shape=[2,1])
y_test = [4.67,7.6]
predict_res = model.predict(x_test, batch_size=32)

# 检查测试集是否在直线上 - R方
y_pred = predict_res.reshape(2).tolist()
print("y_test", y_test, "y_pred", y_pred)
cc = r2_score(y_test,   y_pred)
print('r方 ', cc)

# 绘图
plt.scatter(X, y)
plt.plot(X, model(X), c='r')
plt.show()
print("\nend\n")
