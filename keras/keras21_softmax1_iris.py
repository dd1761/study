from sklearn.datasets import load_iris  # 꽃잎의 길이와 넓이, 줄기의 길이를 가지고 어떤 꽃인지를 맞추는 알고리즘
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)   #input=4 output=1       #pandas .describe() /   .info()
# print(datasets.feature_names)                   #pandas .columns


x = datasets.data
y = datasets['target']
# print(x)
# print(y)
# print(x.shape)  # (150, 4)
# print(y.shape)  # (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,   # False의 문제점은 하나의 데이터가 몰려있어서 예측할 때에 제대로 된 성능이 나오지 않는다.
    random_state=333,   # 분류에서 특정 데이터의 값을 배제하여 계산할 수 있기 때문에 데이터의 균형자체가 무너질 수 있다.
    test_size=0.9,
    stratify=y  # 데이터의 비율을 맞춰줌. ex) 0이 90프로와 1이 10프로인 데이터에서 썼을 때 테스트 사이즈의 비율에서 0과 1의 비율이 5대5정도로 맞게 비율을 맞춰줌.
                # y형 데이터는 분류 데이터에서만 사용가능. ex) 보스턴이나 캘리포니아 데이터에서는 사용불가
)
print(y_train)
# print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='sigmoid'))      #   회귀형식의 모델구성
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))       # 다중분류에서는 softmax, y의 클래스의 수가 3이므로 Dense(3)으로 만들어준다.
                                                # softmax의 y클래스의 확률은 총 합 100%가 나와야 한다.
                                                # 다중분류에서 마지막 노드는 무조건 softmax를 사용.
                                                
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])                                                

model.fit(x_train, y_train, 
          epochs=10, 
          batch_size=1,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

