import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(datasets.DESCR)   #input=4 output=1       #pandas .describe() /   .info()
print(datasets.feature_names)


y = to_categorical(y)

# print(x.shape, y.shape)     #(1797, 64) (1797,)
# print(np.unique(y, return_counts=True))     #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    train_size=0.8,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)

# scaler = MinMaxScaler()            
scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)

print('y : ', y)
print('y.shape : ', y.shape)   #y.shape :  (1797, 10)
print('y_train : ',y_train)
print('y_test : ',y_test)   #y_test :  [[0. 0. 1. ... 0. 0. 0.]

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(64,)))
model.add(Dense(40, activation='sigmoid'))      #   회귀형식의 모델구성
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])                                                

model.fit(x_train, y_train, 
          epochs=100, 
          batch_size=10,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test) 
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])   # 원래의 y값 
# y_predict = model.predict(x_test[:5])   # 예측한 y값
# print(y_predict)        #one hot encoding된 값이 나오게 된다. 

"""
from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)       #원핫인코딩된 형태가 아니라 소수점자리의 데이터값으로 들어가있다.
y_predict = np.argmax(y_predict, axis=1)    #y_predict의 값을 argmax를 통하여 one-hot encoding되어있는 데이터의 값을 원래 상태로 되돌린다.
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)          #y_test의 값을 one-hot encoding되어있는 상태에서 argmax를 통하여 원래의 데이터 형태로 되돌린다.
print('y_test : ', y_test)
# acc = accuracy_score(y_test, y_predict) # y_test의 값은 원핫인코딩이 되어있는 상태이지만 y_predict의 값은 소수점의 값이기 때문에 비교가 되지 않는다.
# print(acc)

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[5])
plt.show()                                  #이미지는 

"""