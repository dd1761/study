# [실습]
# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

path = 'c:/study/_save/'

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    train_size=0.7,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=123                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)


scaler = MinMaxScaler()            
# scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)                  #(14447, 8) (6193, 8)   

x_train = x_train.reshape(14447, 2, 2, 2)
x_test = x_test.reshape(6193, 2, 2, 2)


#2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(256, (2,2), input_shape=(2,2,2), activation='relu'))
model.add(Dropout(0.2)) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

#2. 모델구성(함수형)                                    #함수형의 장점은 순서대로 실행하는 것이 아닌 input부분만 수정하면 순서상관없이 실행가능하다.
# input1 = Input(shape=(8,))                     
# dense1 = Dense(64, activation='relu')(input1) 
# drop1 = Dropout(0.5)(dense1)   
# dense2 = Dense(56, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)                              
# dense3 = Dense(52, activation='sigmoid')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(40, activation='relu')(drop3)
# dense5 = Dense(28, activation='relu')(dense4)
# dense6 = Dense(16, activation='relu')(dense5)
# dense7 = Dense(12, activation='relu')(dense6)
# dense8 = Dense(8, activation='relu')(dense7)
# dense9 = Dense(4, activation='linear')(dense8)
# output1 = Dense(1, activation='linear')(dense9)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )

import datetime
date = datetime.datetime.now()     
print(date)                         # 2023-01-12 14:57:50.668057
print(type(date))                   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")              # string format time    date를 시간형태가 아닌 문자열로 바꿔준다.
print(date)
print(type(date))                   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        #epoch:04는 숫자 네자리까지  ex) 37번의 값이 제일 좋으면 0037 val_loss는 소수점 4번째 자리까지 나오게 됨.


mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k39_02_' + date + '_' + filename
                      )


model.fit(x_train, y_train, epochs=10000, batch_size=10,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 

# model.save(path + "keras30_ModelCheckPoint3_save_model.h5")     # 가중치와 모델 저장.



# model = load_model(path +'MCP/keras30_ModelCheckPoint1.hdf5')


#4. 평가, 예측
print('========================1. 기본 출력 ============================')


mse, mae = model.evaluate(x_test, y_test)




y_predict = model.predict(x_test)

# print("y_test(원래값) : ", y_test)

from sklearn.metrics import  r2_score        # r2는 수식이 존재해 임포트만 하면 사용할 수 있다.
      # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용

r2 = r2_score(y_test, y_predict)        # R2스코어는 높을 수록 평가가 좋다. RMSE의 값은 낮을 수록 평가가 좋다.
print('mse : ', mse)
print("R2스코어  : ", r2)
