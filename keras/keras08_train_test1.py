import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])    #(10, )
# y = np.array(range(10))                 #(10, )
x_train = np.array([1,2,3,4,5,6,7])     #(7, )
x_test = np.array([8,9,10])             #(3, )   수가 달라도 오류가 안남 (특성이 하나이기 때문에)
y_train = np.array(range(7))
y_test = np.array(range(7,10))


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(55))
model.add(Dense(47))
model.add(Dense(40))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mae", optimizer="adam")
model.fit(x_train,y_train, epochs=2000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)


"""
결과
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(55))
model.add(Dense(47))
model.add(Dense(40))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

epochs=700, batch_size=1

loss :  0.12164243310689926
[11]의 결과 :  [[10.1580515]]



epochs=700, batch_size=1
[11]의 결과 :  [[9.942083]]
"""




