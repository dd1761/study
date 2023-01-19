from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train / 255 # 0~1사이의 값으로 바꿔줌
x_test = x_test / 255   # 0~1사이의 값으로 바꿔줌

print(x_train.shape, y_train.shape)     
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))

model = Sequential()            
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))    
model.add(MaxPooling2D((2, 2)))             
model.add(Conv2D(filters=64, kernel_size=(2,2)))    
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2,2)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
modelpath = '../data/modelcheckpoint/k45_cifar100_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
start_time = datetime.datetime.now()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es, cp])
end_time = datetime.datetime.now()
time = end_time - start_time
print("걸린시간 : ", time)


