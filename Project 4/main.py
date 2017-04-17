## Практическое задание № 3. Многоклассовая классификация и нейронные сети (нейронная сеть)

# Инициализация
import numpy as np
import scipy.io as spi
import matplotlib.pyplot as plt

from displayData import displayData
from computeCost import computeCost
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from gradientDescent import gradientDescent
from predictNN import predictNN

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = spi.loadmat('data.mat')

X = data['X']
y = data['y']

m = X.shape[0]

# Визуализация данных
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel, 10)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ================ Часть 2. Загрузка параметров ==================

print('Часть 2. Загрузка параметров')

# Загрузка параметров
weights = spi.loadmat('weights.mat')

# Архитектура рассматриваемой нейронной сети является следующей
# 1. Число слоев: 3
# 2. Число нейронов во входном слое: 400 (равно числу признаков) без учета компоненты смещения
# 3. Число нейронов в скрытом слое:  25 без учета компоненты смещения
# 4. Число нейронов в выходном слое: 10 (равно числу классов)
# 5. Функция активации нейронов скрытого и выходного слоев: сигмоидная

input_layer_size  = 400 # число нейронов во входном слое
hidden_layer_size =  25 # число нейронов в скрытом слое
num_labels        =  10 # число классов (нейронов в выходном слое)

Theta1 = weights['Theta1'] # размер матрицы 25 x 401
Theta2 = weights['Theta2'] # размер матрицы 10 x 26

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# =========== Часть 3. Вычисление стоимостной функции ============

print('Часть 3. Вычисление стоимостной функции')

# Добавление единичного признака
m = X.shape[0]
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

# Задание параметра регуляризации
lam = 0

# Вычисление значений стоимостной функции
J = computeCost(X, y, num_labels, Theta1, Theta2, lam)

print('Значение стоимостной функции для загруженных параметров модели: {:.4f}'.format(J))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# === Часть 4. Вычисление стоимостной функции с регуляризацией ===

print('Часть 4. Вычисление стоимостной функции с регуляризацией')

# Задание параметра регуляризации
lam = 1

# Вычисление значений стоимостной функции
J = computeCost(X, y, num_labels, Theta1, Theta2, lam)

print('Значение стоимостной функции с регуляризацией для загруженных параметров модели: {:.4f}'.format(J))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ====== Часть 5. Вычисление производной сигмоидной функции ======

print('Часть 5. Вычисление производной сигмоидной функции')

z = np.array([1, -0.5, 0, 0.5, 1])
g = sigmoidGradient(z)

print('Значения производной сигмоидной функции для [1, -0.5, 0, 0.5, 1]:')
print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(g[0], g[1], g[2], g[3], g[4] ))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ============== Часть 6. Инициализация параметров ===============

print('Часть 6. Инициализация параметров')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ============== Часть 7. Обучение нейронной сети ================

print('Часть 7. Обучение нейронной сети')

# Задание параметров обучения
iterations = 1500
alpha = 1
lam = 1

Theta1, Theta2, J_history = gradientDescent(X, y, num_labels, initial_Theta1, initial_Theta2, alpha, iterations, lam)

# Визуализация процесса сходимости
plt.figure()
plt.plot(np.arange(len(J_history)) + 1, J_history, '-b', linewidth = 2)
plt.xlabel('Число итераций');
plt.ylabel('Значение стоимостной функции')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# == Часть 8. Вычисление доли правильных ответов классификатора ==

print('Часть 8. Вычисление доли правильных ответов классификатора')

# Вычисление доли правильных ответов классификатора
p = predictNN(X, Theta1, Theta2)
acc = np.sum((p == y).astype('float64')) / len(y) * 100
print('Доля правильных ответов обученного классификатора = {:.4f}'.format(acc))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ========== Часть 9. Визуализация весов нейронной сети ==========

print('Часть 9. Визуализация весов нейронной сети')

displayData(Theta1[:, 1:], 5)