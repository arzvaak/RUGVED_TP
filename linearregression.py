import pandas as pd
import numpy as np

# reads csv 
boston = pd.read_csv("Boston.csv")

#adds dataframes
X = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
Y = boston['medv']

# here adding a column of ones first (bias x0 term) because mafs. And making Y a column vector

one = np.ones((len(X),1))
X = np.append(one, X, axis=1)

Y = np.array(Y).reshape((len(Y),1))


# eq function to get beta from normal equation.
def eq(X,Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
    
    return beta

# predicting using such beta cuz mafs
def predict(X, beta):
    return np.dot(X, beta)

# not doing test-train split, just output everything. 
beta = eq(X,Y)
predictions = predict(X,beta)

# conclusion - very shitty accuracy. 
print(predictions)
