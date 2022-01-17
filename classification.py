import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
seed = 42
np.random.seed(seed)

# load dataset
dataframe = pd.read_csv("./data/iris2.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# define model structure
def baseline_model():
    model=Sequential()
    model.add(Dense(7,input_dim=4,activation='tanh'))#神经元个数7个，输入数据4维，激活函数tanh
    model.add(Dense(2,activation='sigmoid'))#输出层,3个类别
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#损失函数，编译器，评价
    return model
estimator=KerasClassifier(build_fn=baseline_model,epochs=40,batch_size=1,verbose=1)#包装成了机器学习模型
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0, shuffle=False)
# print("X_train ", X_train)

estimator.fit(X_train, Y_train)
 
# # make predictions
pred = estimator.predict(X_test)
# # print("Y_test ", Y_test)
# # inverse numeric variables to initial categorical labels

# # 输出sklearn 统计结果
transfer_test_Y = np.argmax(Y_test, axis=-1)
# print("pred ", pred)
# print("transfer_test_Y ", transfer_test_Y)

print(classification_report(transfer_test_Y, pred))
# 
# k-fold cross-validate
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("kfold results", results)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
