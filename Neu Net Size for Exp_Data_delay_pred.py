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

for k in range(3):
    if k == 0:
        Inputs = np.concatenate((x_meas[:train_nodes, :], x_meas[2:train_nodes+2, :]), axis = 1)
        
        Test_inp = np.concatenate((x_meas[train_nodes:train_nodes+test_nodes, :],
                             x_meas[train_nodes+2:train_nodes+test_nodes+2, :]), axis = 1)
    else:

        Inputs = np.concatenate((Inputs, x_meas[2*(k+1):train_nodes+2*(k+1), :]), axis = 1)
        
        Test_inp = np.concatenate((Test_inp,
                                   x_meas[train_nodes+2*(k+1):train_nodes+test_nodes+2*(k+1), :]), axis = 1)


Outputs = torques[:train_nodes, :]/50.0
                    
#Test_inp = x_meas[train_nodes:train_nodes+test_nodes, :]
Test_out = torques[train_nodes:train_nodes+test_nodes, :]/50.0

time_test = np.linspace(0, duration_test, test_nodes)

RMS_pred_delay = np.zeros((6, 4))

NeuPar = np.array([4, 8, 12, 26])

for j in range(6):
    for k in range(4):
        # Initialize the constructor
        model = Sequential()
        act = LeakyReLU(alpha=0.3)
        
        # Add an input layer
        model.add(Dense(NeuPar[k], activation = act, input_shape=(24,)))
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
        
        model.load_weights('Model_' + str(j+1) + 'HL_' + str(NeuPar[k]) + 'HN_weight_delay.h5')
    
        predictions = model.predict(Test_inp)
        
        RMS_pred_delay[j, k] = np.sqrt((np.sum((predictions[:, 0] - Test_out[:, 0])**2)/test_nodes 
                    + np.sum((predictions[:, 1] - Test_out[:, 1])**2)/test_nodes 
                    + np.sum((predictions[:, 2] - Test_out[:, 2])**2)/test_nodes)/3.0)

#show_nodes = 1000
#
#fig = plt.figure(figsize=(12, 8))
#ax1 = fig.add_subplot(3,1,1)
#ax1.plot(time_test[:show_nodes], Test_out[:show_nodes, 0]*5000.0, label = 'Test_Data')
#ax1.plot(time_test[:show_nodes], predictions[:show_nodes, 0]*5000.0, label = 'Prediction Data')
#plt.xlabel('time (s)')
#plt.ylabel('ankle torque (Nm)')
#fig.add_subplot(3,1,2)
#plt.plot(time_test[:show_nodes], Test_out[:show_nodes, 1]*5000.0, label = 'Test_Data')
#plt.plot(time_test[:show_nodes], predictions[:show_nodes, 1]*5000.0, label = 'Prediction Data')
#plt.xlabel('time (s)')
#plt.ylabel('knee torque (Nm)')
#fig.add_subplot(3,1,3)
#plt.plot(time_test[:show_nodes], Test_out[:show_nodes, 2]*5000.0, label = 'Test_Data')
#plt.plot(time_test[:show_nodes], predictions[:show_nodes, 2]*5000.0, label = 'Prediction Data')
#plt.xlabel('time (s)')
#plt.ylabel('hip torque (Nm)')



# Initialize the constructor
#model = Sequential()
#act = LeakyReLU(alpha=0.3)
#
## Add an input layer
#model.add(Dense(NeuPar, activation = act, input_shape=(24,)))
##    model.add(Activation(act))

#    # Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))

#    # Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))
#    
##         Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))
#    
#        # Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))
#    
#        # Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))
#    
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))
#    
#    # Add hidden layer
#    model.add(Dense(NeuPar[k], activation = act))
##    model.add(Activation(act))

## Add Output layer 
#model.add(Dense(3, activation='linear'))
#
#model.load_weights('Model_1HL_' + str(NeuPar) + 'HN_weight_delay.h5')
#
#predictions = model.predict(Test_inp)
#
#show_nodes = 1000
#
#fig = plt.figure(figsize=(12, 6))
#
#plt.plot(time_test[:show_nodes], Test_out[:show_nodes, 0]/50.0)
#plt.plot(time_test[:show_nodes], predictions[:show_nodes, 0])