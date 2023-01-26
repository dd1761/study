import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ???

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9]])

y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7,3,1)          # => [[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]], [[5],[6],[7]], [[6],[7],[8]], [[7],[8],[9]]
print(x.shape)  # (7, 3, 1) 

#2. 모델구성

model = Sequential()
# model.add(SimpleRNN(units=64, input_shape=(3,1), activation='relu'))      # 3,1 => 3개씩 잘라서 1개씩 예측
                                    # (N, 3, 1) -> ([batch_size, timesteps, feature]])
                                    

model.add(LSTM(units=10, input_shape=(3,1)))    # model.add(SimpleRNN(units=64, input_shape=(3,1), activation='relu'))


model.add(Dense(5, activation='linear'))
model.add(Dense(1))

model.summary()

#심플RNN
# 10 * (10 + 1 + 1 ) = 120
#units * (feature + bias + unists) = parms

#LSTM
# lstm (10 * (10 + 1 + 1) * 4) = 480

