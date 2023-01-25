from sklearn.datasets import load_iris  # 꽃잎의 길이와 넓이, 줄기의 길이를 가지고 어떤 꽃인지를 맞추는 알고리즘
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense,Input, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical   # one hot encoding을 사용하기 위해 to_categorical을 가지고 와 사용한다.
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)   #input=4 output=1       #pandas .describe() /   .info()
# print(datasets.feature_names)                   #pandas .columns


x = datasets.data
y = datasets['target']



y = to_categorical(y)    # y의 값으로 one-hot encoding을 진행하여 y_ca값을 만듬.
# print(y_ca)
# print(x)
# print(y)
# print(x.shape)  # (150, 4)
# print(y.shape)  # (150,)
# print(y.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=1234,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    test_size=0.2,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)

scaler = MinMaxScaler()            
# scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)

# print('y : ', y)
# print('y.shape : ', y.shape)  
# print('y_train : ',y_train)
# print('y_test : ',y_test)

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)

x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)


#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(2,2,1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))       # 다중분류에서는 softmax, y의 클래스의 수가 3이므로 Dense(3)으로 만들어준다.
                                                # softmax의 y클래스의 확률은 총 합 100%가 나와야 한다.
                                                # 다중분류에서 마지막 노드는 무조건 softmax를 사용.
                                                # 수치화를 하였을 때 조심해야 하는 것은 0,1,2 를 각각 동등한 관계로 만들어주어야 한다. 만들어주지 않으면 1과 2의 가치는 2배차이
                                                # one_hot-encoding 원핫인코딩
                                                # y값의 개수만큼 colum이 늘어남.

                                                
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
                      filepath = filepath + 'k39_07_' + date + '_' + filename
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







