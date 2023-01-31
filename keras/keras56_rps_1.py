# 가위 바위 보

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model, save_model, Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


train_datagen = ImageDataGenerator(
    rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
    # horizontal_flip=True,               # 수평 반전
    # vertical_flip=True,                 # 수직 반전 
    # width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
    # height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
    # rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
    # zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
    # shear_range=0.7,                    # 기울임     0.7만큼 기울임
    # fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      # 0~255 -> 0~1 사이로 변환 스케일링 / 평가데이터는 증폭을 하지 않는 원본데이터를 사용한다.    
)   

                                        # x = (160, 150, 150, 1)  y = (160, )
                                        # np.unique(y) = [0. 1.]   0 : 80개  1 : 80개   0 : 정상  1 : 비정상 
                                
                                
# flow 또는 flow_from_directory
xy_train = train_datagen.flow_from_directory(
    'c:/_data/rps/',             # 폴더 경로 지정
    target_size=(200, 200),             # 이미지 사이즈 지정
    batch_size=1000,                       
    class_mode='binary',              # 수치형으로 변환
    # class_mode='categorical',           # one hot encoding
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 160 images belonging to 2 classes.
)



# print(cat_train)                        # <keras.preprocessing.image.DirectoryIterator object at 0x00000182C6143760>
# print(dog_train)                        # <keras.preprocessing.image.DirectoryIterator object at 0x00000182BE134040>




xy_test = test_datagen.flow_from_directory(
    'c:/_data/test/',                  # 폴더 경로 지정
    target_size=(400, 400),             # 이미지 사이즈 지정
    batch_size=1000,
    # class_mode='binary',                # 수치형으로 변환                       
    class_mode='categorical',                # 수치형으로 변환
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 120 images belonging to 2 classes.
)


# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1])               # y값 출력  [0. 0. 1. 1. 1.]
print(xy_train[0][0].shape)           # (1000, 200, 200, 3)
print(xy_train[0][1].shape)           # (1000, )

print(xy_test[0][0].shape)           # (1000, 400, 400, 3)
print(xy_test[0][1].shape)           # (1000, 2)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (3, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(xy_train, steps_per_epoch=32, epochs=100, validation_data=xy_test, validation_steps=4)
