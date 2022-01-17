import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print("tf version", tf.__version__)

#导入 Fashion MNIST 数据集-
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0
# 打印训练集前25个图片
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=1)

# 测试集准确率评估
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 结果转成概率集
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 查看第一个预测结果的输出
print("predictions[0]", predictions[0])
# 查看第一个预测结果的输出中最大的概率所属的标签
print("np.argmax(predictions[0])", np.argmax(predictions[0]))
#查看第一的真实结果
print("real res ", test_labels[0])
# 预测后画出好看的图片https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn

#即便您只使用一个图像，您也需要将其添加到列表中：
img = test_images[1]
img = (np.expand_dims(img,0))
print(img.shape)

# 预测
predictions_single = probability_model.predict(img)
print(predictions_single)
print("预测所属标签 ", np.argmax(predictions_single[0]))
