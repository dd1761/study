from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense,Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
# 23-01-09

#1. 데이터
datasets = load_breast_cancer()

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)


x = datasets['data']    
y = datasets['target']
# print(x.shape, y.shape) # (569, 30), (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,
    random_state=333
)

# scaler = MinMaxScaler()            
scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)


#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dropout(0.5)) 
model.add(Dense(52, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))  # 0과 1사이의 값만 뽑아야 하기 때문에 activation을 sigmoid를 사용한다.

##3. 컴파일, 훈련

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
                      filepath = filepath + 'k31_06_' + date + '_' + filename
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