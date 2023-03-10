import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.font_manager as fm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
    horizontal_flip=True,               # 수평 반전
    vertical_flip=True,                 # 수직 반전 
    width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
    height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
    rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
    zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
    shear_range=0.7,                    # 기울임     0.7만큼 기울임
    fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      # 0~255 -> 0~1 사이로 변환 스케일링 / 평가데이터는 증폭을 하지 않는 원본데이터를 사용한다.    
)   

                                        # x = (160, 150, 150, 1)  y = (160, )
                                        # np.unique(y) = [0. 1.]   0 : 80개  1 : 80개   0 : 정상  1 : 비정상 
                                
                                
# flow 또는 flow_from_directory
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',             # 폴더 경로 지정
    target_size=(100, 100),             # 이미지 사이즈 지정
    batch_size=10,                       
    class_mode='binary',                # 수치형으로 변환
    color_mode='grayscale',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',             # 폴더 경로 지정
    target_size=(100, 100),             # 이미지 사이즈 지정
    batch_size=10,                       
    class_mode='binary',                # 수치형으로 변환
    color_mode='grayscale',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
    # Found 120 images belonging to 2 classes.
)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(100, 100, 1)))     # 99 99 128
model.add(Dropout(0.2))     
model.add(Conv2D(64, (2,2), activation='relu'))             # 98 98 64
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))             # 95 95 32
model.add(Flatten())                                        # 288000
model.add(Dense(16, activation='relu'))                     # 16
model.add(Dense(1, activation='sigmoid'))                   # 0~1 사이의 값으로 출력
# model.add(Dense(2, activation='softmax'))                   # 0~1 사이의 값으로 출력

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',   #val_loss가 10번이상 향상되지 않으면 멈춤.
                              restore_best_weights=True,                        
                              verbose=1 
                              )

# model.compile(loss='binary_crossentropy', optimizer='adam', 
#               metrics=['acc'])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100,
                    validation_data=xy_test, validation_steps=4, callbacks=es)        #validation_steps=4는 120개의 데이터를 10개씩 4번 돌린다는 의미이다.

accuracy = hist.history['acc']                                   
val_acc = hist.history['val_acc']                           
loss = hist.history['loss']                                 
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])

# 그림그리기 plt 땡겨서 

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
path = "C:/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf"
# font_path = "C:/Windows/Fonts/malgun.ttf"
fontprop = fm.FontProperties(fname = path, size = 18)
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)



plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'], color='red', marker='.', label='loss')   # 선의 색은 color='red'빨간색 maker='.'은 선의 형태는 점선으로 label='loss'는 선의 이름은 loss
plt.plot(hist.history['val_loss'], color='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')    #plt의 x축의 이름
plt.ylabel('loss')      #plt의 y축의 이름
plt.title('generator 손실함수', fontproperties=fontprop)
# plt.legend()
plt.legend(loc='upper center')    #upper, lower, center
plt.show()