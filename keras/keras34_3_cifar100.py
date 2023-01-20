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

x_train = x_train / 255 # 0~1사이의 값으로 바꿔줌
x_test = x_test / 255  # 0~1사이의 값으로 바꿔줌

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)   
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)  

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

model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu'))    # (31, 31, 128)
model.add(MaxPooling2D((2, 2)))                       
model.add(Conv2D(filters=84, kernel_size=(2,2)))    # (30, 30, 64)    flatten -> 57600
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))                      
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (29, 29, 32)    flatten -> 26912
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (28, 28, 32)  flatten -> 25088
model.add(Flatten())
model.add(Dense(32, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
model.add(Dense(100, activation='softmax'))     

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        #epoch:04는 숫자 네자리까지  ex) 37번의 값이 제일 좋으면 0037 val_loss는 소수점 4번째 자리까지 나오게 됨.
 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k34_03_' + date + '_' + filename
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

"""
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))    # (31, 31, 128)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (30, 30, 64)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (29, 29, 64)  
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=16, kernel_size=(2,2)))    # (28, 28, 32)  flatten -> 25088
model.add(Flatten())
model.add(Dense(32, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
                                                    
epochs=100, batch_size=32
model.add(Dense(100, activation='softmax'))
loss :  2.52152919769287
acc :  0.358900010585784    



model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))    # (31, 31, 128)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (30, 30, 64)
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (29, 29, 64)  
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=16, kernel_size=(2,2)))    # (28, 28, 32)  flatten -> 25088
model.add(Flatten())
model.add(Dense(32, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
model.add(Dense(100, activation='softmax'))

epochs=100, batch_size=32
model.add(Dense(100, activation='softmax'))
loss :  2.521529197692871
acc :  0.3589000105857849   
loss :  2.428819179534912
acc :  0.3777000010013580
"""