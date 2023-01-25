from tensorflow.keras.datasets import fashion_mnist # keras.datasets에 있는 fashion_mnist를 불러온다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,  MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # fashion_mnist를 불러온다.

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)  x는 60000, 28, 28, 1이라고도 할 수 있기에 흑백이라고 볼 수 있다.
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(x_train[1000])
# print(y_train[1000])       #5



model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), 
                 padding='same',        # valid
                 activation='relu'))    # (28, 28, 128)
model.add(MaxPooling2D())               # (14, 14, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))    # (26, 26, 64)
model.add(MaxPooling2D())               # (13, 13, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (12, 12, 64)
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (11, 11, 32)
model.add(Flatten())
model.add(Dense(32, activation='relu'))             # (60000, 32)
                                                    
model.add(Dense(10, activation='softmax'))          # (60000, 10)

# model.summary()



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
                      filepath = filepath + 'k35_02_' + date + '_' + filename
                      )



model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])  #metrics에 accuracy가 들어갔기 때문에 loss와 accuracy값이 나옴.
model.fit(x_train, y_train, epochs=100, batch_size=32,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 


#4. 평가, 예측
result = model.evaluate(x_test, y_test)   #evaluate는 loss와 metrics를 반환한다.
print('loss : ', result[0])            
print('acc : ', result[1])


# plt.imshow(x_train[1000], 'gray')       #imshow는 이미지를 보여주는 함수.  x_train[1000]을 보여주는데 흑백으로 보여줘라.
# plt.show()


"""


model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), 
                 padding='same',        # valid
                 activation='relu'))    # (28, 28, 128)
model.add(MaxPooling2D())               # (14, 14, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))    # (26, 26, 64)
model.add(MaxPooling2D())               # (13, 13, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (12, 12, 64)
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (11, 11, 32)
model.add(Flatten())
model.add(Dense(32, activation='relu'))             # (60000, 32)
                                                    
model.add(Dense(10, activation='softmax'))          # (60000, 10)

loss :  0.3274413049221039
acc :  0.8945000171661377

  
  
"""