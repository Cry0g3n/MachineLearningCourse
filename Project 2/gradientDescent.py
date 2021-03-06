import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid

def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """
    
    J_history = []
    m = y.shape[0]
    
    for i in range(num_iters):
        
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию

        theta = theta - 1 / m * alpha * np.transpose((np.transpose(sigmoid(X.dot(theta)) - y)).dot(X))
        
        # ============================================================
        
        J_history.append(computeCost(X, y, theta)) # сохранение значений стоимостной функции
                                                    # на каждой итерации
    
    return theta, J_history