#Conv2D로 바꿈.

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))    # 예상 y = 100, 107

timesteps1 = 5 # x는 4개, y는 1개
timesteps2 = 4

print(a.shape)
print(x_predict.shape)

''''''
def split_x(dataset, timesteps1):                       # timesteps1 = 5 5개씩 자르겠다
    aaa = []
    for i in range(len(dataset)-timesteps1+1):          
        subset = dataset[i:(i+timesteps1)]
        aaa.append(subset)
    return np.array(aaa)                                # np.array(aaa) = (96, 5)

bbb = split_x(a, timesteps1)
print(bbb)
print(bbb.shape)


x = bbb[:, :-1]
y = bbb[:, -1]  # bbb[:,5]와 동일

print(x, y)     
print(x.shape, y.shape)     # (96, 4) (96,)

x = x.reshape(96,2,2)               # (96, 4, 1)    
# print(x.shape)

x_predict = split_x(x_predict, timesteps2)                # split_x를 사용.
print(x_predict)                                        # (7, 4)
print(x_predict.shape)                                    # (7, 4)

x_predict = x_predict.reshape(7,2,2)                      # (7, 4, 1)
print(x_predict.shape)                              # (7, 4, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234)

x_train = x_train.reshape(72,2,2,1)     # (72, 4, 1) -> (72, 2, 2, 1)
x_test = x_test.reshape(24,2,2,1)       # (24, 4, 1) -> (24, 2, 2, 1)
x_predict = x_predict.reshape(7,2,2,1)  # (7, 4, 1) -> (7, 2, 2, 1)


print(x_train.shape, y_train.shape) # (72, 2, 2, 1) (72,)
print(x_test.shape, y_test.shape)     # (24, 2, 2, 1) (24,)
print(x_predict.shape)                 # (7, 2, 2)



#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape=(2,2,1)))        # input_shape=(2,2,1) 2개씩 잘라서 1개씩 예측
model.add(Flatten())                                                        
# model.add(LSTM(64, activation='relu', input_shape=(2,2)))       # input_shape=(4,1) 4개씩 잘라서 1개씩 예측
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='linear'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )
model.fit(x_train, y_train, epochs=2000, batch_size=1, callbacks=[es], validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_predict)
print('[96-106]의 결과 : ', result)


