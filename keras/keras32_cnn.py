from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten   #Conv2D는 2차원



model = Sequential()
                            # input = (60000, 5, 5, 1)
model.add(Conv2D(filters=10, kernel_size=(2,2),          #filter는 안써도 상관 없음. kernel_size는 5,5짜리의 그림을 2,2크기의 그림으로 잘라서 4,4짜리의 그림이 됨.
                 input_shape=(5, 5, 1)))             #(N,4,4, 10) 60000개를 N으로 써도 상관없음       #filter는 4,4크기의 그림 하나가 10개 들어간다. 
                                                     # (batch_size, row. columns, channels) channels는 color batch_size는 훈련횟수로 연산. 



model.add(Conv2D(filters=5, kernel_size=(2,2)))      #(N,3,3,5)            #Dense모양과 연결되어야함.
model.add(Flatten())                                 #(N, 45)       #flatten전의 데이터들은 전부 펴짐
model.add(Dense(units=10))                           #(N, 10)       Conv2D에서 Dense의 기본 데이터는 units로 되어있다.
        #input 은 (batch_size, input_dim)
model.add(Dense(4, activation='relu'))  # 지현, 성환, 건률, 렐루                                  #(N, 1)

model.summary()



'''
    padding -   valid : 유효한 영역만 출력. 출력 이미지는 사이즈 입력 사이즈보다 작다.
                same : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일.

    kernel_size : 2D 컴볼루션 창으 ㅣ높이와 너비를 지정하는 정수 또는 2개 정수의 튜플/목록


    input_shape : 샘플 수를 제외한 입력 형태를 정의. 모델에서 첫 레이어일 때만 정의하면 된다. 
                 (행, 열, 채널 수)로 정의. 흑백영상인 경우 채널이 1, 컬러인 경우 채널은 3으로 설정.

    activation  - linear : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나온다.
                - relu : rectifier 함수, 은익층에 주로 쓰인다.
                - sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰인다.
                - softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰인다.

    strides :   기본적으로 입력 데이터와 필터를 연산할 때는 한 칸씩 이동하면서 연산을 하는데 스트라이드는 입력데이터에 필터를 적용할 때 이동할 간격을 조절해주는 파라미터.
    
    
    
    output_shape : 








5,5,1


'''