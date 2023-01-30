import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense,Input, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']



# print(x.shape, y.shape)                 # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))     #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

##################1. keras tocategorical###############################
"""
y = to_categorical(y)
print(y.shape)      #(581012, 8)
print(type(y))
print(y[:10])
print(np.unique(y[:,0], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,1], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))

print('================================================')
y = np.delete(y, 0, axis=1)                     # 전체 데이터중 0번째 데이터의 1열을 지워버림.
print(y.shape)
print(y[:10])
print(np.unique(y[:,0], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0.], dtype=float32), array([581012], dtype=int64))

"""

##################2. pandas get_dummies ###############################
'''
y = pd.get_dummies(y)             #pandas의 get_dummies
print(y[:10])
print(type(y))                    # <class 'pandas.core.frame.DataFrame'>   판다스에서는 데이터 프레임형태는 자동생성된다. 헤더와 인덱스
                                  #pandas의 데이터형태이기 때문에 텐서플로우에서는 상관없이 훈련되지만 뒤 argmax(y_test, axis=1)의 값이 numpy데이터 형태이기 때문에 pandas의 데이터형태인
                                  #getdummies의 데이터형태를 알아보지 못한다.

# y = np.argmax(y, axis=1)          #pandas의 get_dummies
# y = y.values                      # y의 데이터는 판다스였는데 y.value를 통하여 넘파이의 데이터형태로 바꾸어주어야 한다.
# y = y.to_numpy()                  # y의 데이터는 판다스였는데 y.to_numpy를 통하여 넘파이의 데이터형태로 바꾸어주어야 한다.
                
print(y.shape)


'''
##################3. sklearn의 one_hot encoding ###############################

# print('y : ', type(y))
# 힌트 .values  or  .numpy()    pandas
# one-hot encoding 힌트. toarray()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
print(y.shape)      #(581012,)
y = y.reshape(581012, 1)            #(581012,) => (581012,1)
print(y.shape)
# ohe.fit(y)                          # <class 'scipy.sparse._csr.csr_matrix'>    fit에 y를 집어넣어 y의 가중치 값을 저장한다.
# y = ohe.transform(y)
y = ohe.fit_transform(y)            # ohe.fit(y)와 ohe.transform(y)를 한번에 해주는 코드

y = y.toarray()                     # y의 값은 numpy의 데이터형태로 바꿔준다.
print(type(y))


print(y[:15])
print(type(y))      
print(y.shape)      # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)

scaler = MinMaxScaler()            
# scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)        # (435759, 54) (145253, 54)


x_train = x_train.reshape(435759, 9, 6)
x_test = x_test.reshape(145253, 9, 6)


#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Conv1D(64,2, input_shape=(9,6)))
model.add(Dropout(0.3)) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax')) 

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
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
                      filepath = filepath + 'k51_10_' + date + '_' + filename
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

