import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10))
print(x.shape)

'''
x = x.T # (10,3)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = y.T #(10,2)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))    #input_dim= 의 갯수는 열의 수와 같다. 열(컬럼, 피처, 속성) 행무시 열우선
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9,30,210]])
print('[9, 30, 210]의 예측값 : ', result)

""" 
 epochs= 500 batch_size=1 예측값 : [[10.246388, 1.4901893]]
"""

""" 
2차시도 
model.add(Dense(5, input_dim=3))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(2))

epochs=500 batch_size=1
예측값 : 10.212388, 1.4901893
loss : 0.34712
"""
'''