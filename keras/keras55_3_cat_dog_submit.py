import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
path = 'c:/_data/'                  #./ 현재폴더 /하위폴더 / 하위폴더 /
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


