import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터  
x = np.array([1,2,3,4,5,6,7,8,9,10])    #(10, )
y = np.array(range(10)) #(10, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    #test_size=0.3,                                     #(두개 중 하나만 쓰면 가능, 두개 중 어떤것을 사용해도 같은 결과가 나옴) 
    # shuffle=False,                                    #(shuffle는 기본적으로 True가 탑재되어있다.) 
    random_state=123                                    #(random_state를 사용하지 않으면 사용 할때마다 값이 바뀌게 된다.)
)

# x_train = x[:7]
# x_test = x[7:]
# y_train = y[:7]
# y_test = y[7:]


# print('x_train : ', x_train)
# print('x_test : ', x_test)
# print('y_train : ', y_train)
# print('y_test : ', y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(55))
model.add(Dense(47))
model.add(Dense(40))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
#11

#3. 컴파일, 훈련
model.compile(loss="mae", optimizer="adam")
model.fit(x_train,y_train, epochs=2000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)



'''
model.fit(x_train,y_train, epochs=5000, batch_size=1)
loss :  0.09991153329610825
[11]의 결과 :  [[10.268192]]


model.fit(x_train,y_train, epochs=500, batch_size=1)
loss :  0.018053611740469933
[11]의 결과 :  [[10.069034]]

'''
