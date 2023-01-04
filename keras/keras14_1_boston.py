from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data()
y = dataset.target()

print(x)
print(x.shape)
print(y)
print(y.shape)