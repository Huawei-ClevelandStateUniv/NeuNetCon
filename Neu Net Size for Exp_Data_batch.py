#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:32:35 2018

@author: huawei
"""
import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Activation
import keras
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt

x_meas = np.loadtxt('x_meas.txt')
torques = np.loadtxt('torque.txt')

train_nodes = 5000
test_nodes = 2000
duration_test = 40.0

Inputs = x_meas[:train_nodes, :]
Test_inp = x_meas[train_nodes:train_nodes+test_nodes, :]

Outputs = torques[:train_nodes, :]/50.0
                    
#Test_inp = x_meas[train_nodes:train_nodes+test_nodes, :]
Test_out = torques[train_nodes:train_nodes+test_nodes, :]

time_test = np.linspace(0, duration_test, test_nodes)

NeuPar = np.array([4, 26])

for j in range(3, 4):
    for k in range(2):
        # Initialize the constructor
        model = Sequential()
        act = LeakyReLU(alpha=0.3)
        
        # Add an input layer
        model.add(Dense(NeuPar[k], activation = act, input_shape=(6,)))
    #    model.add(Activation(act))
        if j == 1:
            # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
        elif j == 2:
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
            
        elif j == 3:
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
        elif j ==4:
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
            
        elif j ==5:
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                        # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
                                    # Add hidden layer
            model.add(Dense(NeuPar[k], activation = act))
        
        # Add Output layer 
        model.add(Dense(3, activation='linear'))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
        
        StopCond = [EarlyStopping(monitor='loss', patience=10, verbose=0),]
        
        model.fit(Inputs, Outputs, epochs = 100000,
                  batch_size = 1000, callbacks = StopCond)
        
        model.save_weights('Model_' +str(j+1) + 'HL_' + str(NeuPar[k]) + 'HN_weight.h5')
#
#predictions = model.predict(Test_inp)
#
#show_nodes = 500
#
#fig = plt.figure(figsize=(12, 6))
#
#plt.plot(time_test[:show_nodes], Test_out[:show_nodes]/50.0)
#plt.plot(time_test[:show_nodes], predictions[:show_nodes])