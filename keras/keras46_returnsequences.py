import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ???

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80의 예측값을 출력하시오

print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13,3,1)          

#2. 모델구성

model = Sequential()
model.add(LSTM(units=64, input_shape=(3,1), return_sequences=True))             # (N, 64)     # 3,1 => 3개씩 잘라서 1개씩 예측  
                                                # return_sequences=True : LSTM의 출력을 다음 레이어에 전달
model.add(LSTM(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )
model.fit(x, y, epochs=2000, batch_size=1, callbacks=[es], validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

# y_pred = np.array([50, 60, 70]).reshape(1,3,1)
x_pred = x_predict.reshape(1,3,1)
# result = model.predict(y_pred)
result = model.predict(x_pred)
print('[50,60,70]의 결과 : ', result)



