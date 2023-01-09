from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib
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
model = Sequential()
model.add(Dense(26, input_dim=13, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=5, batch_size=10, validation_split=0.2, verbose=1)  # verbos 값 0이면 결과만 표시 1이면 원래대로 표시 2면 프로그램 진행바 제거 3이상이면 epoch값만 표현


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
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)



plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', marker='.', label='loss')   # 선의 색은 color='red'빨간색 maker='.'은 선의 형태는 점선으로 label='loss'는 선의 이름은 loss
plt.plot(hist.history['val_loss'], color='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')    #plt의 x축의 이름
plt.ylabel('loss')      #plt의 y축의 이름
plt.title('보스톤 손실함수')
# plt.legend()
plt.legend(loc='upper right')    #upper, lower, center
plt.show()


'''
verbose=1
걸린시간 :  13.466532707214355


verbose=0
걸린시간 :  10.89578104019165
'''


