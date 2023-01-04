from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) 

x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    test_size=0.3,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=123                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])   #mae는 평균절대오차 mse는 평균제곱오차. metrics의 리스트는 두개이상 사용 가능
model.fit(x_train, y_train, epochs=200, batch_size=1)                                    #loss는 가중치 갱신을 할 때 영향을 미친다. loss는 훈련에 영향을 미친다.

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# mae :  3.0107457637786865 

# mse :  14.8480224609375  





