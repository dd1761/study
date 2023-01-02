import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.7.4

#1. (정제된) 데이터 / 정제되지 않은 데이터는 정제처리를 해야함.
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. 모델구성 y=wx+b
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)

#4. 평가, 예측
results = model.predict([6])
print('6의 예측값 : ', results)