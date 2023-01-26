import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset)-timesteps+1):
        subset = dataset[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]  # bbb[:,5]와 동일

print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)

x = x.reshape(6,4,1) 
print(x.shape)  # (6, 4, 1)

# #2. 모델구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(4,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(5, activation='linear'))
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

# y_pred = np.array([50, 60, 70]).reshape(1,3,1)
x_predict = np.array([7,8,9,10]).reshape(1,4,1)
# result = model.predict(y_pred)
result = model.predict(x_predict)
print('[7,8,9,10]의 결과 : ', result)




