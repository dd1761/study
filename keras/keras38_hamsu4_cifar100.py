from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
#컬러 데이터


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))   
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)   

x_train = x_train / 255
x_test = x_test / 255

#2. 모델구성
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(3,3),padding='same',
#                  input_shape=(32, 32, 3), activation='relu'))    # (31, 31, 128)
# model.add(MaxPooling2D()) 
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))    # (30, 30, 64)  
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (28, 28, 32)  flatten -> 25088
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))             #input_shape = (40000)
#                                                     # (60000, 40000)    (batch_size, input_dim)
# model.add(Dense(10, activation='softmax'))

#2 모델구성(함수형)
input1 = Input(shape=(32*32*3,))                     
dense1 = Dense(512, activation='relu')(input1) 
drop1 = Dropout(0.5)(dense1)   
dense2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)                              
dense3 = Dense(128, activation='sigmoid')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(64, activation='relu')(drop3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(64, activation='relu')(dense5)
dense7 = Dense(128, activation='relu')(dense6)
output1 = Dense(100, activation='linear')(dense7)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',   #val_loss가 10번이상 향상되지 않으면 멈춤.
                              restore_best_weights=True,                        
                              verbose=1 
                              )

date = datetime.datetime.now()              #날짜를 자동으로 저장하기 위해 datetime을 import함.
date = date.strftime("%m%d_%H%M")           #날짜를 원하는 형식으로 바꿔줌.

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        #epoch:04는 숫자 네자리까지  ex) 37번의 값이 제일 좋으면 0037 val_loss는 소수점 4번째 자리까지 나오게 됨.
 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,       #val_loss가 가장 좋은 값이 나올때만 저장하겠다.
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k38_04_' + date + '_' + filename   #filepath에 저장할 경로와 파일명을 지정해줌.
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
loss :  1.20495152473449
acc :  0.591600000858306



model = Sequential()
model.add(Conv2D(filters=200, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu'))    # (31, 31, 128)
model.add(MaxPooling2D((2, 2))) 
model.add(Conv2D(filters=100, kernel_size=(3,3)))    # (30, 30, 64)  
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=70, kernel_size=(2,2)))    # (28, 28, 32)  flatten -> 25088
model.add(Flatten())
model.add(Dense(64, activation='relu'))             #input_shape = (40000)
                                                    # (60000, 40000)    (batch_size, input_dim)
                                                    
epochs=100, batch_size=32,
model.add(Dense(10, activation='softmax'))
loss :  0.95948946475982
acc :  0.67269998788833
    
    
epochs=100, batch_size=32,
model.add(Dense(10, activation='softmax'))
loss :  0.98668211698532
acc :  0.655799984931945
"""