import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']



# print(x.shape, y.shape)                 # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))     #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

##################1. keras tocategorical###############################
"""
y = to_categorical(y)
print(y.shape)      #(581012, 8)
print(type(y))
print(y[:10])
print(np.unique(y[:,0], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,1], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))

print('================================================')
y = np.delete(y, 0, axis=1)                     # 전체 데이터중 0번째 데이터의 1열을 지워버림.
print(y.shape)
print(y[:10])
print(np.unique(y[:,0], return_counts=True))    #y[:,0] 모든 행의 0번째를 보여줌.   (array([0.], dtype=float32), array([581012], dtype=int64))

"""

##################2. pandas get_dummies ###############################
'''
y = pd.get_dummies(y)             #pandas의 get_dummies
print(y[:10])
print(type(y))                    # <class 'pandas.core.frame.DataFrame'>   판다스에서는 데이터 프레임형태는 자동생성된다. 헤더와 인덱스
                                  #pandas의 데이터형태이기 때문에 텐서플로우에서는 상관없이 훈련되지만 뒤 argmax(y_test, axis=1)의 값이 numpy데이터 형태이기 때문에 pandas의 데이터형태인
                                  #getdummies의 데이터형태를 알아보지 못한다.

# y = np.argmax(y, axis=1)          #pandas의 get_dummies
# y = y.values                      # y의 데이터는 판다스였는데 y.value를 통하여 넘파이의 데이터형태로 바꾸어주어야 한다.
# y = y.to_numpy()                  # y의 데이터는 판다스였는데 y.to_numpy를 통하여 넘파이의 데이터형태로 바꾸어주어야 한다.
                
print(y.shape)






'''
##################3. sklearn의 one_hot encoding ###############################

# print('y : ', type(y))
# 힌트 .values  or  .numpy()    pandas
# one-hot encoding 힌트. toarray()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
print(y.shape)      #(581012,)
y = y.reshape(581012, 1)            #(581012,) => (581012,1)
print(y.shape)
# ohe.fit(y)                          # <class 'scipy.sparse._csr.csr_matrix'>    fit에 y를 집어넣어 y의 가중치 값을 저장한다.
# y = ohe.transform(y)
y = ohe.fit_transform(y)            # ohe.fit(y)와 ohe.transform(y)를 한번에 해주는 코드

y = y.toarray()                     # y의 값은 numpy의 데이터형태로 바꿔준다.
print(type(y))


print(y[:15])
print(type(y))      
print(y.shape)      # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    test_size=0.2,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)

scaler = MinMaxScaler()            
# scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(54,)))
model.add(Dense(70, activation='relu'))      #   회귀형식의 모델구성
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])         


from tensorflow.keras.callbacks import EarlyStopping    #대문자면 파이썬의 클래스로 구현되어있다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',  
                              patience=10,                  # earlystopping을 적용하여 최솟값에서 patience값만큼 더 돌다가 최적의값이 갱신이 안되면 멈추게 하는 횟수
                              restore_best_weights=True,    #earlystopping의 값은 최적의 값에서 추가로 몇번 더 돌다가 멈추는데 그 최적의 값을 가지고 오는 메소드
                              verbose=1)                                       

model.fit(x_train, y_train, 
          epochs=5000, 
          batch_size=200,
          validation_split=0.2,
          callbacks=[earlyStopping] ,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test) 
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])   # 원래의 y값 
# y_predict = model.predict(x_test[:5])   # 예측한 y값
# print(y_predict)        #one hot encoding된 값이 나오게 된다. 


y_predict = model.predict(x_test)       #원핫인코딩된 형태가 아니라 소수점자리의 데이터값으로 들어가있다.
y_predict = np.argmax(y_predict, axis=1)    #y_predict의 값을 argmax를 통하여 one-hot encoding되어있는 데이터의 값을 원래 상태로 되돌린다.
print('y_pred(예측값) : ', y_predict[:20])
y_test = np.argmax(y_test, axis=1)          #y_test의 값을 one-hot encoding되어있는 상태에서 argmax를 통하여 원래의 데이터 형태로 되돌린다.
print('y_test(원래값) : ', y_test[:20])
# acc = accuracy_score(y_test, y_predict) # y_test의 값은 원핫인코딩이 되어있는 상태이지만 y_predict의 값은 소수점의 값이기 때문에 비교가 되지 않는다.
# print(acc)


'''

epochs=10, 
batch_size=100,
loss :  0.44826364517211914
accuracy :  0.8096951246261597


epochs=15, 
batch_size=200,
loss :  0.44152364134788513
accuracy :  0.8145142793655396


test_size=0.2,
epochs=5000, 
batch_size=200,
loss :  0.28568756580352783
accuracy :  0.88493412733078

scaler = MinMaxScaler()   
loss :  1.7854512929916382
accuracy :  0.4074421525001526

'''