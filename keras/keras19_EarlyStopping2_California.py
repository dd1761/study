# [실습]
# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  # (20640, 8) input_dim=8
# print(y)
# print(y.shape)  # (20640, ) 

x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    train_size=0.9,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=123                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)

#print(x_train.shape)
#print(dataset.feature_names)    #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

#print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=20, validation_split=0.25, callbacks=[earlyStopping] , verbose=1) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score        # r2는 수식이 존재해 임포트만 하면 사용할 수 있다.
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용

print('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)        # R2스코어는 높을 수록 평가가 좋다. RMSE의 값은 낮을 수록 평가가 좋다.
print("R2 : ", r2)

print('=======================================================')
print(hist) 
print('=======================================================')
print(hist.history) 
print('=======================================================')
print(hist.history['loss'])
print('=======================================================')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt



plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', marker='.', label='loss')   # 선의 색은 color='red'빨간색 maker='.'은 선의 형태는 점선으로 label='loss'는 선의 이름은 loss
plt.plot(hist.history['val_loss'], color='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')    #plt의 x축의 이름
plt.ylabel('loss')      #plt의 y축의 이름
plt.title('california loss')
# plt.legend()
plt.legend(loc='upper right')    #upper, lower, center
plt.show()

'''

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.fit(x_train, y_train, epochs=1000, batch_size=20, validation_split=0.25) 
RMSE :  0.665312120358980
R2 :  0.6823353489455448
'''




# 위 train size 0.9 batch 20 500 중간 마지막 보스턴