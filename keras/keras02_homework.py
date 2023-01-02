import numpy as np
#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [13] : 예측


#2. 모델구성 
from tensorflow.keras.models import Sequential #Sequential 순차적모델을 만듬
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(1, input_dim=1)) #input_dim은 위에 데이터 하나를 통으로 받아들이는 것 ex([1~10]). []리스트로 묶여있는 것 한 덩어리 Dense레이어 사용 input은 x, 1은 y

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') #mae를 사용함. optimizer='adam 로스를 최적화시키는 것에 아담을 이용함.
model.fit(x, y, epochs=3000) #x데이터와 y데이터를 훈련시킴, epochs는 훈련시키는 횟수, 할 때마다 시작값이 달라서 에측값이 달라짐. 최적의 수치도 찾아야함.

#4. 평가, 예측
result = model.predict([13]) #13의 데이터값을 예측
print('결과 : ', result) #결과산출