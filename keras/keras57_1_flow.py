import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 100


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
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x_train[0]을 100개 만들어서 x_data에 넣어라  , x, 전체데이터 -1
    np.zeros(augument_size),                                                  # y_train[0]을 100개 만들어서 y_data에 넣어라, x
    batch_size=augument_size,                                                 # batch_size
    shuffle=True,
)

print(x_data[0]) 
print(x_data[0][0].shape)       # (100, 28, 28, 1)
print(x_data[0][1].shape)       # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off') 
    plt.imshow(x_data[0][0][i], cmap='gray')                            
plt.show()






