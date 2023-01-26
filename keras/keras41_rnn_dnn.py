import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ???

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9]])

y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape) # (7, 3) (7,)

# x = x.reshape(7,3,1)          # => [[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]], [[5],[6],[7]], [[6],[7],[8]], [[7],[8],[9]]
# print(x.shape)  # (7, 3, 1) 

#2. 모델구성

model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3,1), activation='relu'))
model.add(Dense(64, input_shape=(3,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
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

y_pred = np.array([8,9,10]).reshape(1,3,1)
result = model.predict(y_pred)
print('[8,9,10]의 결과 : ', result)

