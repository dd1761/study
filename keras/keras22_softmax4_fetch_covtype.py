import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

y = pd.get_dummies(y)
# y = np.argmax(y, axis=1)    

# print(x.shape, y.shape)                 # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))     #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

# print('y : ', type(y))
# 힌트 .values  or  .numpy()    pandas
# one-hot encoding 힌트. toarray()
y = y.values    # y에 pd.get_dummies(y)로 돌린 값을 y.values를 통해 다시 y의 값만 출력함. 


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    test_size=0.2,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(54,)))
model.add(Dense(70, activation='relu'))      #   회귀형식의 모델구성
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])                                                

model.fit(x_train, y_train, 
          epochs=15, 
          batch_size=200,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test) 
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])   # 원래의 y값 
# y_predict = model.predict(x_test[:5])   # 예측한 y값
# print(y_predict)        #one hot encoding된 값이 나오게 된다. 


y_predict = model.predict(x_test)       #원핫인코딩된 형태가 아니라 소수점자리의 데이터값으로 들어가있다.
y_predict = np.argmax(y_predict, axis=1)    #y_predict의 값을 argmax를 통하여 one-hot encoding되어있는 데이터의 값을 원래 상태로 되돌린다.
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)          #y_test의 값을 one-hot encoding되어있는 상태에서 argmax를 통하여 원래의 데이터 형태로 되돌린다.
print('y_test : ', y_test)
# acc = accuracy_score(y_test, y_predict) # y_test의 값은 원핫인코딩이 되어있는 상태이지만 y_predict의 값은 소수점의 값이기 때문에 비교가 되지 않는다.
# print(acc)

