from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
#컬러 데이터


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
x_train = x_train.reshape(50000 , 32 *32*3)
x_test = x_test.reshape(10000 , 32 *32*3)

x_train = x_train/255.
x_test = x_test/255.


print(np.unique(y_train, return_counts=True)) 

'''
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
'''

#2. 모델구성
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(32 * 32 * 3, )))   # (27, 27, 128) # 28 * 28 = 784
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))             # (60000, 32)
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))             # (60000, 32)                                                    
model.add(Dense(100, activation='softmax'))          # (60000, 10) 

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',   #patience=20, mode='min'  20번까지는 봐주겠다. 21번째부터는 봐주지 않겠다.
                              restore_best_weights=True,                        
                              verbose=1 
                              )

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        #epoch:04는 숫자 네자리까지  ex) 37번의 값이 제일 좋으면 0037 val_loss는 소수점 4번째 자리까지 나오게 됨.
 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,   #mode='auto'는 자동으로 min, max를 구분해줌.
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k36_04_' + date + '_' + filename   #k34_03_1015_1530_0001-1.0000.hdf5
                      )



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])  #metrics에 accuracy가 들어갔기 때문에 loss와 accuracy값이 나옴.
model.fit(x_train, y_train, epochs=200, batch_size=32,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

"""
model.add(Dense(512, activation='relu', input_shape=(32 * 32 * 3, )))   # (27, 27, 128) # 28 * 28 = 784
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))             # (60000, 32)
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))             # (60000, 32)                                                    
model.add(Dense(100, activation='softmax'))          # (60000, 10) 

loss :  3.8090145587921143
acc :  0.11909999698400497

"""