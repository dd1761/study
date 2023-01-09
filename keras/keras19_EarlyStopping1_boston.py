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
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping    #대문자면 파이썬의 클래스로 구현되어있다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',  patience=10, restore_best_weights=True, verbose=1)   #loss값과 val_loss값은 최소값이 가장 좋지만 accuracy 값은 최대값이 좋다.
hist = model.fit(x_train, y_train, epochs=3000, batch_size=10, validation_split=0.2,callbacks=[earlyStopping] , verbose=1)  # verbos 값 0이면 결과만 표시 1이면 원래대로 표시 2면 프로그램 진행바 제거 3이상이면 epoch값만 표현




#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('=======================================================')
print(hist) #<keras.callbacks.History object at 0x000001390FB9C610>
print('=======================================================')
print(hist.history) #hist안에 history라는 변수명이 존재함.  history는 딕셔너리 방식으로 되어있음.
print('=======================================================')
print(hist.history['loss'])
print('=======================================================')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', marker='.', label='loss')   # 선의 색은 color='red'빨간색 maker='.'은 선의 형태는 점선으로 label='loss'는 선의 이름은 loss
plt.plot(hist.history['val_loss'], color='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')    #plt의 x축의 이름
plt.ylabel('loss')      #plt의 y축의 이름
plt.title('boston loss')
# plt.legend()
plt.legend(loc='upper right')    #upper, lower, center
plt.show()


'''
verbose=1
걸린시간 :  13.466532707214355


verbose=0
걸린시간 :  10.89578104019165
'''


