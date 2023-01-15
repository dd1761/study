import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)  x는 60000, 28, 28, 1이라고도 할 수 있기에 흑백이라고 볼 수 있다.
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)    

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,) 
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))    # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))    # (27, 27, 128)
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (26, 26, 64)
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (25, 25, 64)  flattne -> 40000
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (25, 25, 64)  flattne -> 40000
model.add(Flatten())
model.add(Dense(16, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))



#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")       # 월, 일, 시간, 분

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        #epoch:04는 숫자 네자리까지  ex) 37번의 값이 제일 좋으면 0037 val_loss는 소수점 4번째 자리까지 나오게 됨.
 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k34_01_' + date + '_' + filename
                      )



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])  #metrics에 accuracy가 들어갔기 때문에 loss와 accuracy값이 나옴.
model.fit(x_train, y_train, epochs=100, batch_size=32,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])



# es, mcp 적용 / val 적용


'''

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))    # (27, 27, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (25, 25, 64)  flattne -> 40000
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (25, 25, 64)  flattne -> 40000
model.add(Flatten())
model.add(Dense(32, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))


loss :  0.1134900152683258
acc :  0.9692999720573425


'''


