import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time


#1. 데이터
path = './_data/ddarung/'                  #./ 현재폴더 /하위폴더 / 하위폴더 /
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    #pd.read_csv('./_data/_ddarung/train.csv', index_col=0) 이걸 path로 줄인 것.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)      # (1459, 10) input_dim=10 count포함 10개 count를 포함하고 있어 count를 분리해줘야함. 사실상 input_dim=9개 
print(submission.shape)     # (715, 1)

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.info())         #0 hour 1459 1 hour_def_temperature 1457 결측치가 2개
                                # 결측치가 있는 데이터 처리법 : 1번째 : 결측치가 있는 데이터를 뺀다.
# print(test_csv.info())
# print(train_csv.describe())

#-------------------- 결측치 처리 1. 제거   -----------------------#

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)                              # (1328, 10)        1459개에서 null값을 빼서 1328개의 데이터만 남았다.

x = train_csv.drop(['count'], axis=1)   #[1459 rows x 9 columns]으로 만듬 x데이터에서 count라는 항목 하나를 뺀다.
print(x)        #[1459 rows x 9 columns]
y = train_csv['count']
# print(y)
# print(y.shape)      # (1459,)




x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1234
)

print(x_train.shape, x_test.shape)  # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (929,) (399,)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_csv = scaler.fit_transform(test_csv)
         

#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dropout(0.5)) 
model.add(Dense(52, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))

#2. 모델구성(함수형)                                    #함수형의 장점은 순서대로 실행하는 것이 아닌 input부분만 수정하면 순서상관없이 실행가능하다.
# input1 = Input(shape=(9,))                     
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
                      filepath = filepath + 'k31_04_' + date + '_' + filename
                      )


model.fit(x_train, y_train, epochs=10000, batch_size=10,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 


#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!!
# 결측치 때문에!!!
# 결측치가 존재해서 nan이 나온다.


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용
rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)      

from sklearn.metrics import  r2_score        # r2는 수식이 존재해 임포트만 하면 사용할 수 있다.
      # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용

r2 = r2_score(y_test, y_predict)        # R2스코어는 높을 수록 평가가 좋다. RMSE의 값은 낮을 수록 평가가 좋다.
print('mse : ', mse)
print("R2스코어  : ", r2)

# 제출할 데이터
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)   # (715, 1)


#   .to_csv()를 사용해서 submission_0105.csv를 완성하시오


# print(submission)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path + 'submission_01050251.csv')

# CPU 걸린시간 : 
# GPU 걸린시간 :


'''


model.fit(x_train, y_train, epochs=200, batch_size=30)
RMSE :  56.25066190178348

model.fit(x_train, y_train, epochs=500, batch_size=32)
RMSE :  55.80784577742137

train_size=0.7,
model.fit(x_train, y_train, epochs=1000, batch_size=20) 
RMSE :  53.975557256023976




model.fit(x_train, y_train, epochs=500, batch_size=10)
RMSE :  53.583538633337845
model.add(Dense(30, input_dim=9))
model.add(Dense(29))
model.add(Dense(28))
model.add(Dense(27))
model.add(Dense(26))
model.add(Dense(25))
model.add(Dense(24))
model.add(Dense(23))
model.add(Dense(22))
model.add(Dense(21))
model.add(Dense(20))
model.add(Dense(19))
model.add(Dense(18))
model.add(Dense(17))
model.add(Dense(16))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(1))


train_size=0.7,
batch_size=1
scaler =StandardScaler()
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(52))
model.add(Dense(40))
model.add(Dense(28))
model.add(Dense(14))
model.add(Dense(2))
model.add(Dense(1))
RMSE :  53.428193503908375
걸린시간 :  15.852965116500854


train_size=0.7,
scaler = MinMax()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dense(52, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))
RMSE :  52.17764920412936


train_size=0.7,
scaler = MinMax()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dense(52, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))
RMSE :  48.15700299114179

train_size=0.9,
scaler = MinMax()
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dense(52, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))
RMSE :  47.66720040122072

train_size=0.7,
scaler = MinMax()
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dense(52, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))
RMSE :  47.15917122068126
'''