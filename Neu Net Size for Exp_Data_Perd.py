#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:32:35 2018

@author: huawei
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

#import plotly
#plotly.tools.set_credentials_file(username='HuaweiWang',
#                                  api_key='u3JPHaMgbggGVFml1yUd')
#
#import plotly.plotly as py
#import plotly.graph_objs as go

import matplotlib.pyplot as plt

x_meas = np.loadtxt('x_meas.txt')
torques = np.loadtxt('torque.txt')

train_nodes = 5000
test_nodes = 2500
duration_test = 50.0

Inputs = x_meas[:train_nodes, :]
Outputs = torques[:train_nodes, :]

Test_inp = x_meas[train_nodes:train_nodes+test_nodes, :]
Test_out = torques[train_nodes:train_nodes+test_nodes, :]

time_test = np.linspace(0, duration_test, test_nodes)

NeuPar = 18

# Initialize the constructor
model = Sequential()
act = LeakyReLU(alpha=0.3)

# Add an input layer
model.add(Dense(NeuPar, activation = act, input_shape=(6,)))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))


# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add hidden layer
model.add(Dense(NeuPar, activation = act))
#    model.add(Activation(act))

# Add Output layer 
model.add(Dense(3, activation='linear'))

model.load_weights('Model_8HL_18HN_weight.h5')

predictions = model.predict(Test_inp)

show_nodes = 500

fig = plt.figure(figsize=(12, 6))

plt.plot(time_test[:show_nodes], Test_out[:show_nodes, 2]/50.0)
plt.plot(time_test[:show_nodes], predictions[:show_nodes, 2])


#trace1 = go.Scatter(
#    x = time_test,
#    y = Test_out[:, 0],
#    mode = 'markers'
#)
#
#trace2 = go.Scatter(
#    x = time_test,
#    y = Test_out[:, 1],
#    mode = 'markers'
#)
#
#trace3 = go.Scatter(
#    x = time_test,
#    y = Test_out[:, 2],
#    mode = 'markers'
#)
#
#trace4 = go.Scatter(
#    x = time_test,
#    y = predictions[:, 0],
#    mode = 'markers'
#)
#
#trace5 = go.Scatter(
#    x = time_test,
#    y = predictions[:, 1],
#    mode = 'markers'
#)
#
#trace6 = go.Scatter(
#    x = time_test,
#    y = predictions[:, 2],
#    mode = 'markers'
#)
#
#
#data2 = [trace1, trace2, trace3, trace4, trace5, trace6]
#fig2 = go.Figure(data=data2)
#
#py.iplot(fig2, filename='fit_test')
#
#
#
#
#
#
