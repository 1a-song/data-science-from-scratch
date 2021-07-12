# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# 손실함수
# - MSE
# - CEE

# +
import numpy as np

#MSE 
#0.5 를 곱하는 이유; https://en.wikipedia.org/wiki/Delta_rule

def mean_squared_error(y, t):  #y는 신경망의 출력, t는 정답 레이블
    return 0.5 * np.sum((y-t)**2)

#손글씨 숫자 구분하기

#정답은 2 (one-hot-encoding으로 나타낸 숫자 2)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 

#0~9까지 숫자일 확률을 나타내는 배열 (y = softmax 함수의 출력)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] 

print(mean_squared_error(np.array(y), np.array(t)))

# +
#신경망의 출력이 7에서 가장 높다면... 오차가 더 크게 나타남 

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] 
print(mean_squared_error(np.array(y), np.array(t)))


# +
#CEE 

def cross_entropy_error(y,t): 
    delta = 1e-7   # np.log() 함수에 0을 대입하면 -inf로 계산을 진행할 수 없어 delta를 더한다
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] 
cross_entropy_error(np.array(y), np.array(t))

# +
#마찬가지로 정답이 아닌 7을 대입해보면

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] 
print(cross_entropy_error(np.array(y), np.array(t)))

# +
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# +
#mini-batch: 60000개 데이터 중 10개 무작위로 추출해 테스트하기

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) 
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# -

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


# +
#one-hot-encoding이 아닌 숫자 레이블로 주어졌을 때

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


# -

def numerical_diff(f, x):
    h = 1e-4  #0.0001
    return (f(x+h) - f(x-h)) / (2*h)  #중앙 차분을 사용


# +
def function_1(x):
    return 0.01*x**2 + 0.1*x

import matplotlib.pyplot as plt
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()
# -

#x == 5, 10에 대한 미분 계산
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))


# +
def function_2(x):
    return x[0]**2 + x[1]**2

#2개 변수에 대한 편미분, x[0] == 3, x[2] == 4 일때 x[0]에 대한 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0**3.0

print(numerical_diff(function_tmp1, 3.0))

#2개 변수에 대한 편미분, x[0] == 3, x[2] == 4 일때 x[1]에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1
print(numerical_diff(function_tmp2, 4.0))


# +
#기울기: a collection of partial derivates

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  #값 복원
    
    return grad


# +
def function_2(x):
    return x[0]**2 + x[1]**2

#f_2 함수의 3점에서의 기울기 구하기
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
