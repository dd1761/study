# 와인을 감정하는 데이터
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)    #   (178, 13) (178,)
# print(y)
print(np.unique(y))            # 라벨의 unique 한 값 //  y는[0 1 2]만 있다.
print(np.unique(y, return_counts=True)) #   (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,             
    random_state=333, 
    test_size=0.2,
    stratify=y 
)

# scaler = MinMaxScaler()            
scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)


#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dropout(0.5)) 
model.add(Dense(52, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=30, restore_best_weights=True,
                              verbose=1) 
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1500, batch_size=32,
          callbacks=[earlyStopping],
          validation_split=0.2,
          verbose=1)

from sklearn.metrics import accuracy_score
import numpy as np

y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                  # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)                     # 소수점 들어가는 실수 형태로 구성// error 발생
print(acc)
