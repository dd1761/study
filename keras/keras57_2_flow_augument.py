import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()        # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size)   # 0~59999 중에서 augument_size개로 랜덤으로 뽑아라

print(randidx)          # [13668 47940 18360 ... 17561 10630 33654]
print(len(randidx))     # 40000

x_augument = x_train[randidx].copy()     # x_train의 randidx를 복사해서 x_augument에 넣어라, 원본을 보존하기 위해 copy()를 사용한다.
y_augument = y_train[randidx].copy()     # y_train의 randidx를 복사해서 y_augument에 넣어라, 원본을 보존하기 위해 copy()를 사용한다.
print(x_augument.shape, y_augument.shape)                  # (40000, 28, 28) (40000,)

x_augument = x_augument.reshape(40000, 28, 28, 1)



train_datagen = ImageDataGenerator(
    rescale=1./255,                     # 0~255 -> 0~1 사이로 변환 스케일링
    horizontal_flip=True,               # 수평 반전
    # vertical_flip=True,                 # 수직 반전 
    width_shift_range=0.1,              # 좌우 이동  0.1만큼 이동
    height_shift_range=0.1,             # 상하 이동  0.1만큼 이동
    rotation_range=5,                   # 회전       5도까지 회전 최대 회전 각은 180도
    # zoom_range=1.2,                     # 확대       원래 사이즈의 1.2배까지
    shear_range=0.7,                    # 기울임     0.7만큼 기울임
    fill_mode='nearest'                 # 빈자리를 채워줌  nearest: 가장 가까운 값으로 채움 
)


                  
# flow 또는 flow_from_directory
x_augumented = train_datagen.flow(      # flow from directory는 폴더에서 가져오는 것이고 flow는 데이터에서 수치화 된 것을 가져오는 것이다.
    x_augument, 
    y_augument,
    batch_size=augument_size,                                                 # batch_size
    shuffle=True,
)

print(x_augumented[0][0].shape)         # (40000, 28, 28, 1)
print(x_augumented[0][1].shape)         # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)


##














