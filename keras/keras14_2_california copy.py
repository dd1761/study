# [실습]
# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  # (20640, 8) input_dim=8
# print(y)
# print(y.shape)  # (20640, ) 

x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    train_size=0.9,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=123                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)

#print(x_train.shape)
#print(dataset.feature_names)    #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
#print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=10) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score        # r2는 수식이 존재해 임포트만 하면 사용할 수 있다.
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용

print('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)        # R2스코어는 높을 수록 평가가 좋다. RMSE의 값은 낮을 수록 평가가 좋다.
print("R2 : ", r2)

'''
model.add(Dense(10, input_dim=8))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))4
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.fit(x_train, y_train, epochs=150, batch_size=2)
loss :  [0.6231525540351868, 0.57588130235672]
RMSE :  0.7894000789883663
R2 :  0.5287320375789096

model.fit(x_train, y_train, epochs=200, batch_size=20) 
loss :  [0.6139920353889465, 0.5703482031822205]
RMSE :  0.783576320860465
R2 :  0.5356598978855484


train_size=0.7,
model.fit(x_train, y_train, epochs=300, batch_size=20)
loss :  [0.6077343821525574, 0.584107518196106]
RMSE :  0.7795733019342688
R2 :  0.5403920835902903


train_size=0.9,
model.fit(x_train, y_train, epochs=300, batch_size=15) 
loss :  [0.6338421106338501, 0.584040641784668]
RMSE :  0.7961420505888598
R2 :  0.5451175873335106

train_size=0.9
model.fit(x_train, y_train, epochs=300, batch_size=10)
loss :  [0.6341466307640076, 0.5916329026222229]
RMSE :  0.7963332430832786
R2 :  0.5448990822414606

'''




# 위 20 아래 18