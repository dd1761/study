import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

# [실습] 자르기
# train_test_split으로 자르기

x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.35,
    shuffle=False
)
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test,
    test_size=0.4,
    shuffle=False
)
# x_train = np.array(range(1,11)) #   (1~10)
# y_train = np.array(range(1,11)) #   (1~10)
# x_test = np.array([11,12,13])   #   (11~13)
# y_test = np.array([11,12,13])   #   (11~13)
# x_validation = np.array([14,15,16]) #(14~16)
# y_validation = np.array([14,15,16]) #(14~16)

# x_train = x[:10]
# x_test = x[10:13]
# y_train = y[:10]
# y_test = y[10:13]
# x_validation = x[13:]
# y_validation = y[13:]

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)



'''
#2. 모델 
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5000, batch_size=1,
          validation_data=(x_validation, y_validation))     #   x에 대한 예상문제를 평가하는 과정을 추가 (validation_data)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)
'''
