import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    target_size=(200, 200),             # 이미지 사이즈 지정
    batch_size=10,                       
    class_mode='binary',                # 수치형으로 변환
    color_mode='grayscale',             # 흑백으로 변환
    shuffle=True,                       # 데이터를 섞어준다. 파이썬에서는 함수(괄호)안에서 ,를 마지막에 찍어도 작동이 된다.    
)

# Found 160 images belonging to 2 classes.

# test_generator = test_datagen.flow_from_directory(
#     '../data/image/brain/test',
#     target_size=(150, 150),
#     batch_size=5,
#     class_mode='binary'
# )
