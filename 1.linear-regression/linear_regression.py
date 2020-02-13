import numpy as np


class MyLinearRegression:

    def fit_gd(self, X_train, y_train, learning_rate=0.01, n_iters=1e4, epsilon=1e-8):
        # 损失函数
        def J(w, X, y):
            return np.sum((X.dot(w) - y) ** 2) / (2 * len(X))

        # 求偏导
        def dJ(w, X, y):
            return X.T.dot(X.dot(w) - y) / len(X)

        # 梯度下降法
        def gradient_descent(initial_w, X, y):
            i_iter = 0
            w = initial_w.copy()
            while i_iter < n_iters:
                gradient = dJ(w, X, y)
                last_w = w
                w = w - learning_rate * gradient
                if abs(J(w, X, y) - J(last_w, X, y)) < epsilon:
                    break
                i_iter += 1
            return w

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_w = np.zeros(X_b.shape[1])
        self.w_ = gradient_descent(initial_w, X_b, y_train)
        return self
