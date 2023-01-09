from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#2023-01-09

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)  #(506, 13)
print(y.shape)  #(506, )

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,
    random_state=333
)

#2. 모델구성
model = Sequential()
#model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13,)))     #input_shape=() 는 다차원에서 사용.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=3)  # verbos 값 0이면 결과만 표시 1이면 원래대로 표시 2면 프로그램 진행바 제거 3이상이면 epoch값만 표현
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('걸린시간 : ', end - start)



'''
verbose=1
걸린시간 :  13.466532707214355


verbose=0
걸린시간 :  10.89578104019165
'''


