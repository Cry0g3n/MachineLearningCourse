import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent


def learningCurve(X, y, X_val, y_val, alpha, num_iters, lam):
    """
        Функция позволяет выполнить ошибку на обучающем (X, y) 
        и проверочном (X_val, y_val) множествах данных. Вычисленные 
        ошибки необходимы для построения кривых обучения. Здесь 
        alpha - параметр сходимости, num_iters - число итераций 
        градиентного спуска, а lam - параметр регуляризации
    """

    m = y.shape[0]

    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])

    for i in range(m):
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить вычисление ошибок на обучающем и проверочном 
        # множествах данных. При реализации программного кода требуется выполнить 
        # обучение и оценить ошибку обучения на первых i тренировочных примерах, 
        # то есть (X[0:i + 1, :], y[0:i + 1, :]). При вычислении ошибки на проверочном 
        # множестве данных требуется использовать проверочное множество целиком, 
        # то есть (X_val, y_val) на каждом этапе вычисления ошибки

        theta = np.zeros([X.shape[1], 1])
        theta, _ = gradientDescent(X[0:i + 1, :], y[0:i + 1, :], theta, alpha, num_iters, lam)
        error_train[i, :] = computeCost(X[0:i + 1, :], y[0:i + 1, :], theta, 0)
        error_val[i, :] = computeCost(X_val, y_val, theta, 0)

        # ============================================================

    return error_train, error_val
