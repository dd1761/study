import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'                  #./ 현재폴더 /하위폴더 / 하위폴더 /
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    #pd.read_csv('./_data/bike/train.csv', index_col=0) 이걸 path로 줄인 것.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

#print(train_csv)
#print(train_csv.shape)      # (10886, 11)
#print(sampleSubmission.shape)   #(6493, 1)

#print(train_csv.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')

'''
#print(train_csv.info())
#print(test_csv.info())
#print(train_csv.describe())

#print(test_csv.shape)   #(6493, 8)


#-------------------- 결측치 처리 1. 제거   -----------------------#
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)  #(10886, 11)

x = train_csv.drop(['count','casual','registered'], axis=1)   # [10886 rows x 9 columns]으로 만듬 x데이터에서 count라는 항목 하나를 뺀다.
print(x)                                # [10886 rows x 9 columns]
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,
    shuffle=True,
    random_state=1234
)

print(x_train.shape, x_test.shape)  
print(y_train.shape, y_test.shape)  

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(13,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(10,activation='relu'))
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
hist = model.fit(x_train, y_train, epochs=1500, batch_size=15, validation_split=0.25, callbacks=[earlyStopping] , verbose=1)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
#1
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
sampleSubmission['count'] = y_submit
# print(sampleSubmission)

sampleSubmission.to_csv(path + 'sampleSubmission_0106.csv')

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
plt.title('bike loss')
# plt.legend()
plt.legend(loc='upper right')    #upper, lower, center
plt.show()


'''

'''