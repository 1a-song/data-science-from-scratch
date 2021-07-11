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

# activation function: 입력 신호의 총합을 출력 신호로 변환하는 함수 (활성화를 일으키는지 결정하는 역할)
# - step function
# - sigmoid function
# - ReLU function
# 이같은 비선형 함수로 은닉층을 구성한다

import numpy as np
import matplotlib.pyplot as plt


# +
#step function

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
#x는 실수(부동소수점)만 가능 (즉, np.array를 인수로 넣을 수 없음) -> 수정하면

def step_function(x):
    y = x > 0  #boolean y
    return y.astype(np.int)

#step function graph

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.show()


# +
#sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# +
#ReLU (Rectified Linear Unit) ; 입력이 0 이하이면 0으로 출력, 0 이상은 그대로 출력

def relu(x):
    return np.maximum(0, x)


# +
#신경망의 내적 

X = np.array([1, 2])
W = np.array([[1, 3, 5],
             [2, 4, 6]])

print(X.shape)
print(W.shape)

Y = np.dot(X, W)
print(Y)
# -

# 3층 신경망 구현하기
# - 0층: 입력층 X = [x1, x2]
# - 1층: 첫번째 은닉층 A1 = [a1, a2, a3] (a^(1))
# - 2층: 두번째 은닉층 A2 = [a1, a2]  (a^(2))
# - 3층: 출력층 Y = [y1, y2]

# +
#각 층의 신호전달 구현하기

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],
              [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1) #활성화 함수 h()로 변환된 신호 Z

print(A1)
print(Z1)

# +
W2 = np.array([[0.1, 0.4], 
               [0.2, 0.5],
               [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2)
print(Z2)


# +
def identity_function(x): #항등함수
    return x

W3 = np.array([[0.1, 0.3],
              [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) #or Y = A3
print(Y)


# +
#구현 정리
def init_network(): #초기화
    network = {} 
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x): #입력 신호를 출력으로 변환하는 처리과정 구현
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


# -

# 출력층 설계하기
# - 회귀 (regression) : identity function
# - 분류 (classification) : softmax function

# +
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

#overflow 방지를 위해서 최대값(c)을 빼준다
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

#softmax() 함수를 사용한 신경망의 출력
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y)) #출력 총합은 1이다. 이런 특징으로 출력을 '확률'로 해석 가능하다.

#Note: 추론 단계에서는 출력층의 softmax 함수를 생략하는 것이 일반적 (신경망 학습에서는 출력층에서 softmax 함수 사용)
