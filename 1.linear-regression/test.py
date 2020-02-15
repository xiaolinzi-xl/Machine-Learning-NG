import numpy as np
import pandas as pd 
from linear_regression import MyLinearRegression
from sklearn.linear_model import LinearRegression

def featule_scale(X_train):
    n = X_train.shape[1]
    res = np.empty(X_train.shape,dtype=float)
    arr_mean = np.array([np.mean(X_train[:, i]) for i in range(n)])
    arr_scale = np.array([np.std(X_train[:, i]) for i in range(n)])
    for i in range(n):
        res[:,i] = (X_train[:,i] - arr_mean[i]) / arr_scale[i]
    return res

if __name__ == "__main__":
    df = pd.read_table(r'C:\code\PycharmProjects\Machine-Learning-NG\1.linear-regression\ex1data2.txt',sep=',',names=['X1','X2','y'])
    X1 = np.array(df['X1'])
    X2 = np.array(df['X2'])
    y = np.array(df['y'])

    X1 = X1.reshape(len(X1),-1)
    X2 = X2.reshape(len(X2),-1)
    X = np.hstack([X1,X2])

    X_trained_standard = featule_scale(X)

    my_lin_reg = MyLinearRegression()
   
    # 使用梯度下降法 
    w1 = my_lin_reg.fit_gd(X_trained_standard,y)
    print(f'梯度下降法的解为 {w1}')

    # 使用正规方程解
    w2 = my_lin_reg.fit_normal(X_trained_standard,y)
    print(f'正规方程的解为 {w2}')

    # 调用 sklearn 中的线性回归
    lin_reg = LinearRegression()
    lin_reg.fit(X_trained_standard,y)
    print('sklearn 中的解为',lin_reg.intercept_,lin_reg.coef_)