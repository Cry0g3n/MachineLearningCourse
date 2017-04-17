import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def gradientDescent(X, y, num_labels, Theta1, Theta2, alpha, num_iters, lam):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели Theta1 и Theta2, используя матрицу 
        объекты-признаки X, вектор меток y, число классов num_labels, 
        параметр сходимости alpha, число итераций алгоритма num_iters 
        и параметр регуляризации lam
    """
    
    J_history = []
    m = y.shape[0]

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    Y = np.zeros((m, num_labels))
    for c in range(num_labels):
        Y[np.where(y == c)[0], c] = 1
    
    for i in range(num_iters):

        print('Эпоха обучения №', i + 1)
        
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить алгоритм обратного распространения ошибки 
        # с целью поиска частных производных от стоимостной функции по 
        # параметрам модели
        
        
        
        # ============================================================
        
        Theta1 = Theta1 - alpha * Theta1_grad
        Theta2 = Theta2 - alpha * Theta2_grad
        
        J_history.append(computeCost(X, y, num_labels, Theta1, Theta2, lam)) # сохранение значений стоимостной функции
                                                                             # на каждой итерации
    
    return Theta1, Theta2, J_history