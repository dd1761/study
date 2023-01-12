from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten   #Conv2D는 2차원

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2),          #filter는 안써도 상관 없음. kernel_size는 5,5짜리의 그림을 2,2크기의 그림으로 잘라서 4,4짜리의 그림이 됨.
                 input_shape=(5, 5, 1)))                #filter는 4,4크기의 그림 하나가 10개 들어간다. 
model.add(Conv2D(filters=5, kernel_size=(2,2)))          #Dense모양과 연결되어야함.
model.add(Flatten())                                    #flatten전의 데이터들은 전부 펴짐
model.add(Dense(10))
model.add(Dense(1))

model.summary()



