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

# print(train_csv)
# print(train_csv.shape)      # (1459, 10) input_dim=10 count포함 10개 count를 포함하고 있어 count를 분리해줘야함. 사실상 input_dim=9개 

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.info())         #0 hour 1459 1 hour_def_temperature 1457 결측치가 2개
                                # 결측치가 있는 데이터 처리법 : 1번째 : 결측치가 있는 데이터를 뺀다.
# print(test_csv.info())
# print(train_csv.describe())

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

print(x_train.shape, x_test.shape)  # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)  # (1021,) (438,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!!
# 결측치 때문에!!!
# 결측치가 존재해서 nan이 나온다.

'''
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용
rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

# 제출할 데이터
y_submit = model.predict(test_csv)
'''
