import pandas as pd
import numpy as np

boston = pd.read_csv("Boston.csv")

X = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
Y = boston['medv']

one = np.ones((len(X),1))
X = np.append(one, X, axis=1)

Y = np.array(Y).reshape((len(Y),1))

def eq(X,Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
    
    return beta

def predict(X, beta):
    return np.dot(X, beta)

beta = eq(X,Y)
predictions = predict(X,beta)

print(predictions)
