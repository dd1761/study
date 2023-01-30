import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터

x1_datasets = np.array([range(100), range(301, 401)]).transpose()   # (2, 100)  나는 100행 2열을 원함! transpose를 사용하면 된다.
print(x1_datasets.shape)                                            # (100, 2)  #삼성전자 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()  # (3, 100) 나는 100행 3열을 원함! transpose를 사용하면 된다.
print(x2_datasets.shape)                                            # (100, 3)  #아모레 시가, 고가, 종가
x3_datasets = np.array([range(100, 200), range(1301, 1401)])

y = np.array(range(2001, 2101))                                     # (100, )   #삼성전자 종가
y2 = np.array(range(201, 301))                                      # (100, )   #아모레 종가
                                  
print(y.shape)

print(x3_datasets.shape)
x3_datasets = x3_datasets.reshape(100, 2)                           # x3데이터를 reshape를 사용하여 100행 2열로 바꿔준다.
print(x3_datasets.shape)                                            # (100, 2)  


x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,\
    y_train, y_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, y2, train_size=0.7, shuffle=True, random_state=1234)


#2. 모델구성
input1 = Input(shape=(2,))                                          
dense1 = Dense(256, activation='relu', name='ds11')(input1)          # input1을 받아서 dense1을 만들고, 그것을 dense2에 넣는다.
dense2 = Dense(128, activation='relu', name='ds12')(dense1)          
dense3 = Dense(64, activation='relu', name='ds13')(dense2)
dense4 = Dense(32, activation='relu', name='ds14')(dense3)
output1 = Dense(14, activation='relu', name='ds15')(dense4)


input2 = Input(shape=(3,))
dense21 = Dense(256, activation='linear', name='ds21')(input2)
dense22 = Dense(128, activation='linear', name='ds22')(dense21)
dense23 = Dense(64, activation='linear', name='ds23')(dense22)
dense24 = Dense(32, activation='linear', name='ds24')(dense23)
output2 = Dense(23, activation='linear', name='ds25')(dense24)

input3 = Input(shape=(2,))
dense31 = Dense(256, activation='linear', name='ds31')(input3)
dense32 = Dense(128, activation='linear', name='ds32')(dense31)
dense33 = Dense(64, activation='linear', name='ds33')(dense32)
dense34 = Dense(32, activation='linear', name='ds34')(dense33)
output3 = Dense(33, activation='linear', name='ds35')(dense34)

from tensorflow.keras.layers import concatenate  # concatenate : 병합하는 함수a
merge1 = concatenate([output1, output2, output3], name='mg1')          # concatenate : 병합하는 함수b
merge2 = Dense(128, activation='relu', name='mg2')(merge1)              
merge3 = Dense(64, name='mg3')(merge2)
merge4 = Dense(32, name='mg4')(merge3)
merge5 = Dense(16, name='mg5')(merge4)
last_output = Dense(1, name='last')(merge5)                 # 마지막 노드1은 y의 shape와 동일해야 한다.

# 2-5. 모델 분기1
dense41 = Dense(256, activation='relu', name='ds41')(last_output)
dense42 = Dense(128, activation='relu', name='ds42')(dense41)
dense43 = Dense(64, activation='relu', name='ds43')(dense42)
dense44 = Dense(32, activation='relu', name='ds44')(dense43)
output4 = Dense(33, activation='linear', name='ds45')(dense44)

# 2-6. 모델 분기2
dense51 = Dense(256, activation='relu', name='ds51')(last_output)
dense52 = Dense(128, activation='relu', name='ds52')(dense51)
dense53 = Dense(64, activation='relu', name='ds53')(dense52)
dense54 = Dense(32, activation='relu', name='ds54')(dense53)
output5 = Dense(33, activation='linear', name='ds55')(dense54)

model = Model(inputs=[input1, input2, input3], outputs=[output4, output5]) # 모델 선언, 모델의 시작은 input1, input2이고 마지막은 last_output이다.

model.summary()

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',   #val_loss가 10번이상 향상되지 않으면 멈춤.
                              restore_best_weights=True,                        
                              verbose=1 
                              )


model.compile(loss='mse', optimizer='adam')  #metrics에 accuracy가 들어갔기 때문에 loss와 accuracy값이 나옴.
model.fit([x1_train, x2_train, x3_train], [y_train, y2_train], epochs=1000, batch_size=1,
          callbacks=[es],
          verbose=1,
          validation_split=0.2
          ) 

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y_test, y2_test])       
print('loss : ', loss)


'''

loss :  [612.7621459960938, 10.915609359741211, 601.8465576171875]  






'''