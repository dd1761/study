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
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=10, batch_size=7) #데이터 나누기 batch_size 가중치 W는 생성되어있음.

#4. 평가, 예측
loss = model.evaluate(x, y) #평가는 evaluate에 x와 y를 넣었을 때 loss값으로 평가한다. evaluate값은 loss값을 반환해준다
#evaluate에 들어가는 값은 훈련데이터가 들어가면 안됨
print('loss : ', loss)
result = model.predict([6]) #통산적으로 predict로 판단하지 않는다. 판단의 기준은 loss로 한다. 
                            #예외적으로 모자이크처리 같은 경우는 사람의 눈으로 판단해야 한다

print('6의 결과 : ', result)
 
 
 
"""
batch_size=1 일때 6/6으로 총 데이터 6개중 1개씩 나누어 6번 훈련 
batch_size=2 일때 3/3으로 총 데이터 6개중 2개씩 나누어 3번 훈련
batch_size=3 일때 2/2으로 총 데이터 6개중 3개씩 나누어 2번 훈련
batch_size=4 일때 2/2으로 총 데이터 6개중 4개씩 나누어 2번 훈련 데이터의 수가 정확히 나누어 떨어지지 않아도 나머지 값으로 훈련 4개/2개
batch_size=6 일때 1/1으로 총 데이터 6개중 6개씩 나누어 1번 훈련 데이터의 수가 일치하여 한번에 훈련
batch_size=7 일때 1/1으로 총 데이터 6개중 7개씩 나누어 1번 훈련 배치사이즈의 수가 더 크면 한번에 훈련

"""