# https://www.kaggle.com/competitions/dogs-vs-cats/data


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, GlobalAveragePooling2D


# train_datagen = ImageDataGenerator(
#     rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
#     # horizontal_flip=True,               # 수평 반전
#     # vertical_flip=True,                 # 수직 반전 
#     # width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
#     # height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
#     # rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
#     # zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
#     # shear_range=0.7,                    # 기울임     0.7만큼 기울임
#     # fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255                      # 0~255 -> 0~1 사이로 변환 스케일링 / 평가데이터는 증폭을 하지 않는 원본데이터를 사용한다.    
# )   

#                                         # x = (160, 150, 150, 1)  y = (160, )
#                                         # np.unique(y) = [0. 1.]   0 : 80개  1 : 80개   0 : 정상  1 : 비정상 
                                
                                
# # flow 또는 flow_from_directory
# xy_train = train_datagen.flow_from_directory(
#     './_data/brain/train/',             # 폴더 경로 지정
#     target_size=(200, 200),             # 이미지 사이즈 지정
#     batch_size=10000,                       
#     class_mode='binary',              # 수치형으로 변환
#     # class_mode='categorical',           # one hot encoding
#     color_mode='grayscale',             # 흑백으로 변환
#     shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
#     # Found 160 images belonging to 2 classes.
# )

# xy_test = test_datagen.flow_from_directory(
#     './_data/brain/test/',             # 폴더 경로 지정
#     target_size=(200, 200),             # 이미지 사이즈 지정
#     batch_size=10000,
#     class_mode='binary',                # 수치형으로 변환                       
#     # class_mode='categorical',                # 수치형으로 변환
#     color_mode='grayscale',             # 흑백으로 변환
#     shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
#     # Found 120 images belonging to 2 classes.
# )

# print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x0000018B20D77AC0>

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)


# # print(xy_train[0])
# # print(xy_train[0][0])
 

# # print(type(xy_train))                # <class 'keras.preprocessing.image.DirectoryIterator'>
# # print(type(xy_train[0]))             # <class 'tuple'>  리스트와 동일하다. 튜플은  한번 생성하면 수정이 불가능하다.
# # print(type(xy_train[0][0]))          # <class 'numpy.ndarray'> numpy로 변경
# # print(type(xy_train[0][1]))          # <class 'numpy.ndarray'> numpy로 변경

# print(xy_train[0][1])               # y값 출력  [0. 0. 1. 1. 1.]
# print(xy_train[0][0].shape)         # (7, 200, 200, 1)
# print(xy_train[0][1].shape)         # (7,)

# np.save('./_data/brain/brain_x_train_x.npy', arr=xy_train[0][0])    # x값 저장  
# np.save('./_data/brain/brain_y_train_x.npy', arr=xy_train[0][1])    # y값 저장  
# # np.save('./_data/brain/brain_xy_train_x.npy', arr=xy_train[0])    # xy값 저장  x, y가 튜플로 묶여있는 상태
 
# np.save('./_data/brain/brain_x_test_x.npy', arr=xy_test[0][0])      # x값 저장
# np.save('./_data/brain/brain_y_test_x.npy', arr=xy_test[0][1])      # y값 저장

x_train = np.load('c:/_data/cat_dog/cat_dog_x_train_x.npy')              # x_train값 불러오기
y_train = np.load('c:/_data/cat_dog/cat_dog_y_train_y.npy')              # y_train값 불러오기
x_test = np.load('c:/_data/train/cat_dog_x_test_x.npy')                # x_test값 불러오기
y_test = np.load('c:/_data/train/cat_dog_y_test_y.npy')                # y_test값 불러오기

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

dog = train_datagen.flow_from_directory(
    'c:/_data/cat_dog/dog/',             # 폴더 경로 지정
    target_size=(200, 200),             # 이미지 사이즈 지정               
    class_mode=None,              # 수치형으로 변환
    # class_mode='categorical',           # one hot encoding
    color_mode='rgb',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 160 images belonging to 2 classes.
)


# xy_test = np.load('./_data/brain/brain_xy_train_x.npy')             # xy_test값 불러오기

# print(x_train.shape, x_test.shape)                                  # (160, 200, 200, 1) (120, 200, 200, 1)
# print(y_train.shape, y_test.shape)                                  # (160,) (120,)
# print(x_test[100])

print(dog[0][0].shape)



#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

hist = model.fit( #xy_train[0][0],xy_train[0][1],    
                    x_train, y_train,
                    batch_size=16, epochs=100,     # xy_train[0][0] = x_train, xy_train[0][1] = y_train
                    validation_data=[x_test, y_test]
                    # validation_steps=25
                    )        #validation_steps=4는 120개의 데이터를 10개씩 4번 돌린다는 의미이다.


accuracy = hist.history['acc']                                   
val_acc = hist.history['val_acc']                           
loss = hist.history['loss']                                 
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])

loss2 = model.predict(dog)

print(loss2)
