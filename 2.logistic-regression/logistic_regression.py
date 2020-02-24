import numpy as np


class MyLogisticRegression:

    def fit_gd(self, X_train, y_train, learning_rate=0.01, n_iters=1e4, epsilon=1e-8):
        def sigmoid(theta, X):
            return 1 / (1 + np.exp(-1 * X.dot(theta)))

        # cost function
        def J(theta, X, y):
            return np.sum(-1 * y * np.log(sigmoid(theta, X)) - (1 - y) * np.log(sigmoid(theta, X))) / len(X)

        # 求偏导
        def dJ(theta, X, y):
            return ((sigmoid(theta, X) - y).T.dot(X)) / len(X)

        # 梯度下降法
        def gradient_descent(initial_theta, X, y):
            i_iter = 0
            theta = initial_theta.copy()
            while i_iter < n_iters:
                gradient = dJ(theta, X, y)
                last_theta = theta
                theta = theta - learning_rate * gradient
                if abs(J(theta, X, y) - J(last_theta, X, y)) < epsilon:
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        theta = gradient_descent(initial_theta, X_b, y_train)
        return theta
