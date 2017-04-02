import numpy as np


def normalEqn(X, y):
    """ 
        Функция позволяет вычислить параметры модели для линейной
        регресии с использованием нормальных уравнений
    """

    theta = 0

    # ====================== Ваш код здесь ======================
    # Инструкция: выполнить вычисление параметров модели для линейной 
    # регрессии с использованием норамаьных уравнений

    theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)

    # ============================================================

    return theta