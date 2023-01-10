import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

#11


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,
    shuffle=True,
    random_state=1234
)

print(x_train.shape, x_test.shape)  # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (929,) (399,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))



#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=10,              
                              restore_best_weights=True,
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=30000, batch_size=1,validation_split=0.25, callbacks=[earlyStopping] , verbose=1)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!!
# 결측치 때문에!!!
# 결측치가 존재해서 nan이 나온다.


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용
rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)      

print('걸린시간 : ', end-start)


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

print('=======================================================')
print(hist) 
print('=======================================================')
print(hist.history) 
print('=======================================================')
print(hist.history['loss'])
print('=======================================================')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', marker='.', label='loss')   # 선의 색은 color='red'빨간색 maker='.'은 선의 형태는 점선으로 label='loss'는 선의 이름은 loss
plt.plot(hist.history['val_loss'], color='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')    #plt의 x축의 이름
plt.ylabel('loss')      #plt의 y축의 이름
plt.title('ddarungy loss')
# plt.legend()
plt.legend(loc='upper right')    #upper, lower, center
plt.show()



'''

RMSE :  85.25672302220333


train_size=0.9,
model.fit(x_train, y_train, epochs=200, batch_size=1,validation_split=0.25)
RMSE :  58.978108924123504

train_size=0.9,
model.fit(x_train, y_train, epochs=300, batch_size=1,validation_split=0.25)
RMSE :  54.89042746781144


train_size=0.9,
model.fit(x_train, y_train, epochs=500, batch_size=1,validation_split=0.25)
RMSE :  53.59895542445503

train_size=0.9,
model.fit(x_train, y_train, epochs=6000, batch_size=1,validation_split=0.25)
RMSE :  48.2372733413749
'''