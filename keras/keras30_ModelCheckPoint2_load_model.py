from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# path = './_save/'
# path = '../_save/'
path = 'c:/study/_save/'

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(    
    x, y,
    train_size=0.8,                                      #train데이터와 test데이터의 비율을 7:3으로 설정
    shuffle=True,                                       #shuffle=True면 랜덤데이터를 사용. shuffle=False면 순차적인 데이터를 사용.
    random_state=1234                                    #random_state는 123번에 저장되어있는 랜덤데이터를 사용. 
                                                        #random_state를 사용하지 않으면 프로그램을 실행할 때마다 값이 달라진다.
)

# scaler = MinMaxScaler()            
scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)




"""
#2. 모델구성(함수형)                                    #함수형의 장점은 순서대로 실행하는 것이 아닌 input부분만 수정하면 순서상관없이 실행가능하다.
input1 = Input(shape=(13,))                     
dense1 = Dense(64, activation='relu')(input1)    
dense2 = Dense(56, activation='relu')(dense1)                              
dense3 = Dense(52, activation='sigmoid')(dense2)
dense4 = Dense(40, activation='relu')(dense3)
dense5 = Dense(28, activation='relu')(dense4)
dense6 = Dense(16, activation='relu')(dense5)
dense7 = Dense(12, activation='relu')(dense6)
dense8 = Dense(8, activation='relu')(dense7)
dense9 = Dense(4, activation='linear')(dense8)
output1 = Dense(1, activation='linear')(dense9)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=100, restore_best_weights=True,
                              verbose=1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath= path +'MCP/keras30_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=10000, batch_size=10,
          callbacks=[es, mcp],
          verbose=1,
          validation_split=0.2,
          ) 

"""

model = load_model(path +'MCP/keras30_ModelCheckPoint1.hdf5')


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




'''
