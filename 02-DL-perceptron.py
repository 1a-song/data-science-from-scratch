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

#AND Gate
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

# +
#theta를 bias로 치환

import numpy as np

x = np.array([0, 1]) #input
w = np.array([0.5, 0.5]) #weight(가중치)
b = -0.7 #bias (뉴런을 활성화시키는 편향값)
w*x
# -

np.sum(w*x)

np.sum(w*x) + b


#weight, bias를 도입한 AND, NAND, OR Gate 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# +
#multi-layer perceptron으로 구현하는 XOR Gate

import pandas as pd

columns = ['x1', 'x2', 's1', 's2', 'y']
truth_table = pd.DataFrame([[0, 0, 1, 0, 0],
                            [1, 0, 1, 1, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 0, 1, 0]], columns=columns)
truth_table.head()


# +
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
# -


