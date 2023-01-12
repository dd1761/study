from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
# 23-01-09

#1. 데이터
datasets = load_breast_cancer()

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)


x = datasets['data']    
y = datasets['target']
# print(x.shape, y.shape) # (569, 30), (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,
    random_state=333
)

# scaler = MinMaxScaler()            
scaler =StandardScaler()
# scaler.fit(x_train)                        # scaler에 대한 x값을 가중치에 저장
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)       #위에 scaler.fit이랑 transform과정을 한번에 적용한 것.
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 0과 1사이의 값만 뽑아야 하기 때문에 activation을 sigmoid를 사용한다.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])    # 지금이해하기는 힘들다. 일단 외워라!


from tensorflow.keras.callbacks import EarlyStopping    #대문자면 파이썬의 클래스로 구현되어있다.

# earlyStopping = EarlyStopping(monitor='val_loss',   # monitor='val_loss'를 쓰는 이유는 val_loss를 사용하는 것이 가장 좋다. loss를 사용해도 괜찮다.
earlyStopping = EarlyStopping(monitor='accuracy',   # monitor='val_loss'를 쓰는 이유는 val_loss를 사용하는 것이 가장 좋다. loss를 사용해도 괜찮다.
                              mode='auto',          # min, max, auto   loss와 val_loss는 min, accuracy값은 max 모르겠으면 auto
                              patience=10, 
                              restore_best_weights=True, 
                              verbose=1)   #loss값과 val_loss값은 최소값이 가장 좋지만 accuracy 값은 최대값이 좋다.

model.fit(x_train, y_train, epochs=10000, 
          batch_size=16, 
          validation_split=0.2, 
          callbacks=[earlyStopping] , # callbacks=[earlyStopping] 모델을 훈련시키고 실행하는 과정에서 가장 최적의 값을 찾아내면 epochs값만큼 실행하지 않고 조기종료 함.
          verbose=1)   

#4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# print('loss, accruacy : ', loss)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss, : ', loss) # loss :  [0.17049993574619293, 0.9473684430122375]     앞에 첫번째 값은 loss값이고 나머지 한개는 metrics=['accuracy'] 값으로 나온다.
print('accuracy : ', accuracy)


y_predict = model.predict(x_test)

# print(y_predict[:10])           # -> 정수형으로 바꾸어야 한다.
# print(y_test[:10])

# y_predict =y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)
print('y_predict : \n', y_predict)


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
