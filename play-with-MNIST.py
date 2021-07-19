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
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
# %matplotlib inline

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes        
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #가중치 업데이트
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), 
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                    np. transpose(inputs))
        
        pass
    
    
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# +
#노드 셋팅, 학습 데이터 준비
input_nodes = 784 #28*28
hidden_nodes = 100
output_nodes = 10
    
learning_rate = 0.3
    
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
training_data_file = open("/Users/wonah/Project/from-scratch/dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#신경망 학습시키기
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

# +
#테스트 데이터 준비하기
test_data_file = open("/Users/wonah/Project/from-scratch/dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close

#테스트 해보기
all_values = test_data_list[0].split(',')
print(all_values[0])

image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
# -

n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

# +
# 신경망 성능 평가하기
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "network's answer")
    
    if (label == correct_label):
        scorecard.append(1)  #정답인 경우 +1
        
    else:
        scorecard.append(0) #오답인 경우 +0
        pass
    
    pass

print(scorecard)

scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum() / scorecard_array.size)
# -

# 지금까지 샘플 데이터를 신경망에 적용시켜 봤다면... 60,000개 레코드를 적용하고 결과 확인해보기

# +
input_nodes = 784 #28*28
hidden_nodes = 100
output_nodes = 10
    
learning_rate = 0.3
    
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
training_data_file = open("/Users/wonah/Project/from-scratch/dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

test_data_file = open("/Users/wonah/Project/from-scratch/dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close

all_values = test_data_list[0].split(',')
print(all_values[0])

image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

# +
# 신경망 성능을 비교해보면...
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01    
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    
    if (label == correct_label):
        scorecard.append(1)  
        
    else:
        scorecard.append(0) 
        pass
    
    pass

scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum() / scorecard_array.size)
# -

# 뜯어보기

data_file = open("/Users/wonah/Project/from-scratch/dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines() #파일이 크다면 readlines는 no (메모리 낭비됨)
data_file.close()

len(data_list)

data_list[0] #5로 레이블된 손글씨 데이터

# +
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28)) #asfarray 문자열을 실수로 변환 (컴퓨터가 읽을 수 있는 실수)
plt.imshow(image_array, cmap='Greys', interpolation='None') #시각화해서 나타내기
# -

# 입력값 준비하기
# 0~255 사이에 색상값 범위(input)를 0.01~1.0 사이로 조정
scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

# +
# 0.01~1.0 사이가 맞는지 검증

import pandas as pd
df = scaled_input
print(df[df<0.01])
print(df[df>1.00])
# -

# 출력 준비하기
# 출력 노드는 0~9, 10개
# 활성화 함수가 도달할 수 없는 0과 1 값을 사용하지 않도록 조정
onodes = 10
targets = np.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
