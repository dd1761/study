from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)               # scaler에 대한 x값을 가중치에 저장
x = scaler.transform(x)


print(x)
print(type(x))              # x의 데이터 타입은 <class 'numpy.ndarray'>

print('최소값 : ',np.min(x))
print('최대값 : ',np.max(x))



x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    train_size=0.7,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=123                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)

#print(x_train.shape)
#print(dataset.feature_names)    #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

#print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(26, input_dim=13, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10000, batch_size=10,
          verbose=1,
          validation_split=0.25) 

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)


y_predict = model.predict(x_test)

print("y_test(원래값) : ", y_test)

from sklearn.metrics import mean_squared_error, r2_score        # r2는 수식이 존재해 임포트만 하면 사용할 수 있다.
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       # np.sqrt는 값에 루트를 적용한다. mean_squared_error은 mse값 적용

print('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)        # R2스코어는 높을 수록 평가가 좋다. RMSE의 값은 낮을 수록 평가가 좋다.
print("R2 : ", r2)


'''
model.add(Dense(26, input_dim=13))
model.add(Dense(52, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='linear'))
model.fit(x_train, y_train, epochs=10000, batch_size=10,validation_split=0.25) 
RMSE :  4.4227262309073
R2 :  0.75799864986511
변환전

변환후

'''
