import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


#1. 데이터
x = np.array([1,2,3,4,5,6]) #데이터를 작게 나눌수록 성능이 좋아짐. ex) [1,2] ,[3,4], [5,6] 작게 나눌수록 시간이 오래걸린다.
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(50))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=10, batch_size=2) #데이터 나누기 batch_size

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)



