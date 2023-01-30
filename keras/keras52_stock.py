import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

PATH = 'c:/study/_data/'

samsung = pd.read_csv(PATH + '삼성전자 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(samsung)
# print(samsung.shape) #(1980, 17)

amore = pd.read_csv(PATH + '아모레퍼시픽 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(amore)
# print(amore.shape)   #(2220, 17)

# 삼성전자 x ,y 추출
samsung_x = samsung[['고가', '저가','종가', '외인(수량)', '기관']]
samsung_y = samsung[['시가']].to_numpy() # x 데이터는 아래에서 split 함수를 거치며 numpy array로 변환되므로 y는 여기서 변환해준다

# print(samsung_x)
# print(samsung_y)
# print(samsung_x.shape) # (1980, 5)
# print(samsung_y.shape) # (1980, 1)

# 아모레 x, y 추출
amore_x = amore.loc[1979:0,['고가', '저가', '종가', '외인(수량)', '시가']]
# print(amore_x)
# print(amore_x.shape) #(1980, 5)

samsung_x = MinMaxScaler().fit_transform(samsung_x)
amore_x = MinMaxScaler().fit_transform(amore_x)

def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

samsung_x = split_data(samsung_x, 5)
amore_x = split_data(amore_x, 5)
# print(samsung_x.shape) #(1976, 5, 5)
# print(amore_x.shape) #(1976, 5, 5)

samsung_y = samsung_y[4:, :] # x 데이터와 shape을 맞춰주기 위해 4개 행 제거
# print(samsung_y.shape) #(1976, 1)

# 예측에 사용할 데이터 추출 (마지막 값)
samsung_x_predict = samsung_x[-1].reshape(-1, 5, 5)
amore_x_predict = amore_x[-1].reshape(-1, 5, 5)
# print(samsung_x_predict.shape) # (5, 5, 1)
# print(amore_x_predict.shape) # (5, 5, 1)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test  = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.7, random_state=333)

print(samsung_x_train.shape, samsung_x_test.shape)  # (1383, 5, 5) (593, 5, 5)
print(samsung_y_train.shape, samsung_y_test.shape) # (1383, 1) (593, 1)
print(amore_x_train.shape, amore_x_test.shape)  # (1383, 5, 5) (593, 5, 5)


# 삼성전자
input_sm = Input(shape=(5, 5))
dense_sm1 = LSTM(1024, return_sequences=True,activation='relu')(input_sm)
dense_sm2 = Dropout(0.2)(dense_sm1)
dense_sm3 = LSTM(512, activation='relu')(dense_sm2)
dense_sm4 = Dense(256, activation='relu')(dense_sm3)
dense_sm5 = Dropout(0.2)(dense_sm4)
dense_sm6 = Dense(128, activation='relu')(dense_sm5)
dense_sm7 = Dense(64, activation='relu')(dense_sm6)
dense_sm8 = Dropout(0.2)(dense_sm7)
dense_sm9 = Dense(32, activation='relu')(dense_sm8)
output_sm = Dense(1)(dense_sm9)

# 아모레퍼시픽
input_am = Input(shape=(5, 5))
dense_am1 = LSTM(1024, return_sequences=True,activation='relu')(input_am)
dense_am2 = Dropout(0.2)(dense_am1)
dense_am3 = LSTM(512, activation='relu')(dense_am2)
dense_am4 = Dense(256, activation='relu')(dense_am3)
dense_am5 = Dropout(0.2)(dense_am4)
dense_am6 = Dense(128, activation='relu')(dense_am5)
dense_am7 = Dense(64, activation='relu')(dense_am6)
dense_am8 = Dropout(0.2)(dense_am7)
dense_am9 = Dense(32, activation='relu')(dense_am8)
output_am = Dense(1)(dense_am9)

# 병합
merge1 = concatenate([output_sm, output_am])
merge2 = Dense(64, activation='relu')(merge1)
merge3 = Dense(128, activation='relu')(merge2)
merge4 = Dense(64, activation='relu')(merge3)
merge5 = Dense(32, activation='relu')(merge4)
output_mg = Dense(1, activation='relu')(merge5)

model = Model(inputs=[input_sm, input_am], outputs=[output_mg])
model.summary()


model.compile(loss='mse', optimizer= 'adam')


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              restore_best_weights=True,                        
                              verbose=1 
                              )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath = path +'MCP/keras30_ModelCheckPoint3.hdf5'
                      filepath = filepath + 'k52_Samsung' + date + '_' + filename
                      )
model.fit([samsung_x_train, amore_x_train], samsung_y_train , epochs=1024, batch_size=128, validation_split=0.2, callbacks=[es, mcp]) 


# model.save_weights(PATH + 'stock_weight.h5') # 가중치 저장

# model = load_model('c:/study/_save/MCP/k52_Samsung0129_2225_0061-1923355264.0000.hdf5')   # 모델 불러오기

loss = model.evaluate([samsung_x_test, amore_x_test], samsung_y_test)

samsung_y_predict=model.predict([samsung_x_predict, amore_x_predict])


print("loss : ", loss)
print("삼성전자 시가 :" , samsung_y_predict)


'''

loss :  1075919744.0
삼성전자 시가 : [[66176.7]]


loss :  2914764544.0
삼성전자 시가 : [[65339.234]]

'''