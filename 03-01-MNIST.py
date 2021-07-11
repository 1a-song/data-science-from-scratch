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

# +
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

#https://github.com/WegraLee/deep-learning-from-scratch/tree/master/dataset

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#Creating pickle file이 출력된다
#프로그램 실행 중 특정 객체를 pickle 파일로 저장하고 pickle 파일을 로드하면 객체가 즉시 복원되어 편리하다

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# +
import numpy as np
from PIL import Image
import sys, os
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #numpy로 저장된 이미지 데이터 -> PIL용 데이터객체로 변환
    pil_img.show()
    
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28) #flatten으로 1차원 넘파이 배열로 저장된 이미지 다시 28x28 크기로 복원
print(img.shape)

img_show(img)


# +
#신경망의 추론 처리 ; 입력층 뉴련 784개 (28x28) , 출력층 10개 (0~9)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


x, t = get_data() #데이터셋 획득
network = init_network() #네트워크 생성

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i]) #분류
    p = np.argmax(y) #확률이 가장 높은 원소의 인덱스
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# +
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)

# +
#Batch processing

x, t = get_data()
network = init_network()

batch_size = 100 #배치 크기
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  #axis = 1, column indexing
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
