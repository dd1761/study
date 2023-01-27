# 일요일까지(일요일 23시59분까지) 과제 제출

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

#1. 데이터

x1_datasets = np.array([range(100), range(301, 401)]).transpose()   # (2, 100)  나는 100행 2열을 원함! transpose를 사용하면 된다.
print(x1_datasets.shape)                                            # (100, 2)  #삼성전자 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()  # (3, 100) 나는 100행 3열을 원함! transpose를 사용하면 된다.
print(x2_datasets.shape)                                            # (100, 3)  #아모레 시가, 고가, 종가

y = np.array(range(2001, 2101))                                     # (100,)    #삼성전자 하루 뒤 종가
print(y.shape)                                                      # (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234          # x데이터를 2개이상 집어넣어도 순서가 바뀌지 않으면 가능하다.
)

print(x1_train.shape, x2_train.shape, y_train.shape)                # (70, 2) (70, 3) (70,)  #train_size=0.7
print(x1_test.shape, x2_test.shape, y_test.shape)                   # (30, 2) (30, 3) (30,)  #train_size=0.7

#2. 모델구성 첫번째 
input1 = Input(shape=(2,))                                          
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2. 모델구성 두번째
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3 모델병합
from tensorflow.keras.layers import concatenate  # concatenate : 병합하는 함수a
merge1 = concatenate([output1, output2], name='mg1')          # concatenate : 병합하는 함수b
merge2 = Dense(10, activation='relu', name='mg2')(merge1) 
merge3 = Dense(13, name='mg3')(merge2)  
last_output = Dense(1, name='last')(merge3)                 # 마지막 노드1은 y의 shape와 동일해야 한다.

model = Model(inputs=[input1, input2], outputs=last_output) # 모델 선언, 모델의 시작은 input1, input2이고 마지막은 last_output이다.

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)




