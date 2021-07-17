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
        
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                    np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                    np.transpose(inputs))                      
        
        pass
    
    
    def query(self, inputs_list):
        
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_inputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
        pass

# +
import numpy as np
import scipy.special

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# -

n.query([1.0, 0.5, -1.5])

# 뜯어보기

# 가중치
np.random.rand(3, 3)  #3x3 행렬로 0~1 사이 난수 생성
np.random.rand(3, 3) - 0.5 #가중치는 음수일 수도 있음, -0.5로 -0.5 ~ 0.5 사이 난수 생성으로 변경

self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

#더 정교한 가중치
#np.random.normal(정규분포 중심, 연결노드 개수에 루크를 씌우고 역수를 취한 표준편차, 행렬 형태)
self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

# +
#은닉계층으로 들어오는 신호 계산
hidden_inputs = np.dot(self.wih, inputs)

import scipy.special
self.activation_function = lambda x: scipy.special.expit(x) #시그모이드 함수 호출

#은닉계층에서 나오는 신호 계산
hidden_outputs = self.activation_function(hidden_inputs)

#최종 출력 계층으로 들어오는 신호를 계산
final_inputs = np.dot(self.who, hidden_outputs)
#최종 출력 계층에서 나가는 신호를 계산
final_outputs = self.activation_function(final_inputs)

# +
#오차 계산 (실제값 - 계산값)
output_errors = targets - final_outputs

#error_hidden = w_ho.T x error_output
#은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차를 재조합
hidden_errors = np.dot(self.who.T, output_errors)

#은닉 계층과 출력 계층간 가중치 업데이트 (hidden-output weight)
self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                             np.transpose(hidden_outputs))

#입력 계층과 은닉 계층간 가중치 업데이트 (input-hidden weight)
self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                             np.transpose(inputs)
