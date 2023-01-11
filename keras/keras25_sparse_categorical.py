from sklearn.datasets import load_iris  # 꽃잎의 길이와 넓이, 줄기의 길이를 가지고 어떤 꽃인지를 맞추는 알고리즘
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical   # one hot encoding을 사용하기 위해 to_categorical을 가지고 와 사용한다.
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)   #input=4 output=1       #pandas .describe() /   .info()
# print(datasets.feature_names)                   #pandas .columns


x = datasets.data
y = datasets['target']



# y = to_categorical(y)    # y의 값으로 one-hot encoding을 진행하여 y_ca값을 만듬.
# print(y_ca)
# print(x)
# print(y)
# print(x.shape)  # (150, 4)
# print(y.shape)  # (150,)
# print(y.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    test_size=0.2,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)
print('y : ', y)
print('y.shape : ', y.shape)  
print('y_train : ',y_train)
print('y_test : ',y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='sigmoid'))      #   회귀형식의 모델구성
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))       # one_hot encoding은 하지 않았지만 자동으로 형을 변형해준다 y의 노드의 개수는 3개.

                                                
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])                                                

model.fit(x_train, y_train, 
          epochs=100, 
          batch_size=1,
          validation_split=0.2,
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
print('y_pred(예측값) : ', y_predict)

# y_test = np.argmax(y_test, axis=1)          # one_hot encoding을 하지 않았으니 쓸 필요가 없다. 
print('y_test(원래값) : ', y_test)
acc = accuracy_score(y_test, y_predict) # y_test의 값은 원핫인코딩이 되어있는 상태이지만 y_predict의 값은 소수점의 값이기 때문에 비교가 되지 않는다.
print(acc)





#시작할 때 sparse_categorical_entorpy를 확인하여 one_hot encoding을 하였는지 확인하고 스파스 카테고리 엔트로피가 존재한다면 원핫인코딩이 안되어있고 있으면 원핫이 되어있는 코드이다.

