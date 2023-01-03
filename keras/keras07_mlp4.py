import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10)])
# print(range(10))
x = x.T

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = y.T #(10,2)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) #input_dim= 의 갯수는 열의 수와 같다. 열(컬럼, 피처, 속성) 행무시 열우선
model.add(Dense(1000))
model.add(Dense(900))
model.add(Dense(800))
model.add(Dense(700))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=20)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([9])
print('[9]의 예측값 : ', result)

""" 
 epochs= 1000 batch_size=1 
 [9]예측값 : [[10,026596, 1.6176409, -0.0763025]]
 loss : 0.09640394
"""

""" 
2차시도 
model.add(Dense(5, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(3))

epochs= 5000 batch_size=20 
 [9]예측값 : [[9.92078, 1.6344309, -0.08081061]]
 loss : 0.09640394
"""

"""
model.add(Dense(5, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(3))

model.fit(x, y, epochs=2000, batch_size=10)
[9]예측값 : [[10.038808, 1.4863151, 0.18130341]]
loss : 0.0868
"""