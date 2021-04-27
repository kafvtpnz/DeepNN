# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2021

@author: AM4
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Считываем данные 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')

# df.to_csv('data_err.csv', index=None)
    
# возьмем перые 100 строк, 4-й столбец 
y = df.iloc[0:100, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, 0)

# возьмем два признака, чтобы было удобне визуализировать задачу
X = df.iloc[0:100, [0, 2]].values

# добавим фиктивный признак для удобства матричных вычслений
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

# Признаки в X, ответы в y - постмотрим на плоскости как выглядит задача (y==1)
plt.figure
plt.scatter(X[0:50, 1], X[0:50, 2], color='red', marker='o')
plt.scatter(X[50:100, 1], X[50:100, 2], color='blue', marker='x')

# функцию нейрора из первой работы дополним функцией активации:
# значение = f(w1*признак1+w2*признак2+w0)
# f = 1/(1-exp(-x))

# начальные значения весов зададим пока произвольно
w = np.array([0, 0.1, 0.4])
# сам нейрон запишем в виде одной строки
predict = 1/(1+np.exp(-np.dot(X[1], w)))


# gradient decent    
N=len(X)
# инициализируем веса случайными числами, задаем скорость обучения 
# и параметр сглаживания функционала качества
w=np.random.random((X.shape[1]))
nu = 0.5
lambd = 1

costs=[0] # тут будем хранить информацию об ошибке для построения графика
w_iter = []

for i in range(100): # пока просто сделаем какое-то количество итераций
    
    # делаем начальное предсказание 
    y_pred = 1/(1+np.exp(-np.dot(X, w)))
    
    #и оценку функционала качества обучения
    cost = (1/N)*np.sum((y_pred-y)**2)
    
    #вычисляем градиент
    dw=(2/N)*np.dot((y_pred-y),X)
    
    #обновляем веса
    w = w - nu * dw
    #записываем результат в список
    costs.append((1-lambd)*costs[-1]+lambd*cost)   

# строим результат на графике
plt.figure
plt.plot(costs)

# вычислим ошибку на обучающей выборке
y_pred = 1/(1+np.exp(-np.dot(X, w)))
print(sum(abs((y_pred>0.5)-y)))

# и на полной выборке
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == "Iris-setosa", 1, 0)
X_all = df.iloc[:, [0, 2]].values
X_all = np.concatenate([np.ones((len(X_all),1)), X_all], axis = 1)
y_pred = 1/(1+np.exp(-np.dot(X_all, w)))
print(sum(abs((y_pred>0.5)-y_all)))


# посмотрим как выглядит зависимость функционала ошибки от значений весов

# создадим сетку параметров (весов)
w1 = np.arange(-2,2,0.1)
w2 = np.arange(-2,2,0.1)
W1, W2 = np.meshgrid(w1, w2)

# напишем функцию вычисления функционала ошибок так, чтобы она работала на сетке 
# w0 будет зафиксирован, w1 и w2 изменяются
def cost_func(X, y, W1, W2):
    cost = np.zeros(W1.shape)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            y_pred = 1/(1+np.exp(-np.dot(X, np.array([-1, W1[i,j], W2[i,j]]))))
            cost[i,j] = (1/N)*np.sum((y_pred-y)**2)
    return cost


# строим зависимость в 3d
from mpl_toolkits import mplot3d
С = cost_func(X, y, W1, W2) # ситаем функционал ошибк на сетке весов
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(W1, W2, С, color='green') # строим "каркас"
ax.plot_surface(W1, W2, С, rstride=1, cstride=1,
                cmap='winter', edgecolor='none') # делаем заливку поверхности
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('cost')
plt.show()

# stochastic gradient decent    
N=len(X)
w=np.random.random((X.shape[1]))
nu = 1
lambd = 1

costs=[0.5]
w_iter = []

for i in range(2): # теперь корректировку веса делаем после каждого примера
    # перебираем примеры в выборке, но т.к. примеры нужно выбирать случайно 
    # перемешаем выборку 
    new_ind = np.random.permutation(N)
    X = X[new_ind]
    y = y[new_ind]
    for x,ytarget in zip(X,y):
        
        y_pred = 1/(1+np.exp(-np.dot(x, w)))
        cost = (1/N)*np.sum(((1/(1+np.exp(-np.dot(X, w))))-y)**2)
        
        #вычисляем градиент
        dw=np.dot((y_pred-ytarget),x)
        
        #обновляем веса
        w = w - nu * dw
        #записываем результат в список
        costs.append((1-lambd)*costs[-1]+lambd*cost)   

# строим результат на графике
plt.figure
plt.plot(costs)

# вычислим ошибку на обучающей выборке
y_pred = 1/(1+np.exp(-np.dot(X, w)))
print(sum(abs((y_pred>0.5)-y)))

# и на полной выборке
y_pred = 1/(1+np.exp(-np.dot(X_all, w)))
print(sum(abs((y_pred>0.5)-y_all)))

