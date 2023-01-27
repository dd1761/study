import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping


a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))    # 예상 y = 100, 107

timesteps1 = 5 # x는 4개, y는 1개

def split_x(dataset, timesteps1):
    aaa = []
    for i in range(len(dataset)-timesteps1+1):
        subset = dataset[i:(i+timesteps1)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps1)
print(bbb)
print(bbb.shape)        # (96, 5)


x = bbb[:, :-1]     # bbb[:,0:4]와 동일
y = bbb[:, -1]      # bbb[:,5]와 동일

print(x, y)     
print(x.shape, y.shape)     # (96, 4) (96,)


# x = x.reshape(96,4,1)
# print(x.shape)  

timesteps2 = 4
def split_x(x_preidct, timesteps2):                      # x_predict를 split_x 함수로 잘라준다.
    aaa = []                                            
    for i in range(len(x_predict)-timesteps2+1):
        subset = x_predict[i:(i+timesteps2)]
        aaa.append(subset)
    return np.array(aaa)                               

x_predict = split_x(x_predict, timesteps2)               # x_predict를 split_x 함수로 잘라준다.
print(x_predict)                                        
print(x_predict.shape)                                  # (7, 4)

"""
x_predict = x_predict.reshape(7,4,1)                    # x_predict를 7,4,1로 reshape
print(x_predict.shape)

# #2. 모델구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(4,1)))       # input_shape=(4,1) 4개씩 잘라서 1개씩 예측
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
model.fit(x, y, epochs=2000, batch_size=1, callbacks=[es], validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

# y_pred = np.array([100,101,102,103,104,105,106]).reshape(1,8,1)
result = model.predict(x_predict)
# result = model.predict(x_predict)
print('[96-106]의 결과 : ', result)
"""