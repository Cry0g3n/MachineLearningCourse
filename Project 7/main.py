## Практическое задание № 7. Анализ главных компонент

# Инициализация
import numpy as np
import scipy.io as spi
import matplotlib.pyplot as plt
from numpy.matlib import repmat

from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from displayData import displayData

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X
data = spi.loadmat('data1.mat')

X = data['X']

# Визуализация данных
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.xlabel('Первый признак')
plt.ylabel('Второй признак')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ============== Часть 2. Анализ главных компонент ===============

print('Часть 2. Анализ главных компонент')

# Выполнение нормализации признаков
X_norm, mu, sigma = featureNormalize(X)

# Вычисление главных компонент
U, S = pca(X_norm)

print('Матрица собственных векторов:')
print(U)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# =============== Часть 3. Сокращение размерности ================

print('Часть 3. Сокращение размерности')

# Задание числа числа сохраненных главных компонент
K = 1

# Выполнение сокращения размерности данных
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

print('Первые три восстановленных примера:')
print(X_rec[0:3, :])

# Визуализация данных
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo', label = 'Исходные данные')
plt.plot(X_rec[:, 0],  X_rec[:, 1],  'rx', label = 'Аппроксимация данных')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Первый признак')
plt.ylabel('Второй признак')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ========== Часть 4. Загрузка и визуализация базы лиц ===========

print('Часть 4. Загрузка и визуализация базы лиц')

# Загрузка данных и формирование матрицы объекты-признаки X
data = spi.loadmat('data2.mat')

X = data['X']

# Визуализация данных
displayData(X[0:100, :], 10)

# Выполнение нормализации признаков
X_norm, mu, sigma = featureNormalize(X)

# Вычисление главных компонент
U, S = pca(X_norm)

# Визуализация собственных лиц
displayData(U[:, 0:36].transpose(), 6)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ======= Часть 5. Сокращение размерности изображений лиц ========

print('Часть 5. Сокращение размерности изображений лиц')

# Задание числа числа сохраненных главных компонент
K = 100

# Выполнение сокращения размерности данных
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

# Визуализация восстановленных данных
X_rec = X_rec * repmat(sigma, X.shape[0], 1) + repmat(mu, X.shape[0], 1)
displayData(X_rec[0:100, :], 10)