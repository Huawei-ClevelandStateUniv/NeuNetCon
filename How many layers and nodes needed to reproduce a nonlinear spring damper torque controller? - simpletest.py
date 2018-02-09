# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import plotly
plotly.tools.set_credentials_file(username='HuaweiWang',
                                  api_key='u3JPHaMgbggGVFml1yUd')

import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

num_nodes = 500
num_states = 3

Inputs = 1 - 2*np.random.random((num_nodes, num_states))
Outputs = np.zeros((num_nodes, num_states))
Test_out = np.zeros((num_nodes, num_states))

K0 = [1000, 600, 200]

Ks = [5000, 3000, 1000]

Outputs[:, 0] = (K0[0] + Ks[0]*abs(Inputs[:, 0])**3)*Inputs[:, 0]
Outputs[:, 1] = (K0[1] + Ks[1]*abs(Inputs[:, 1])**3)*Inputs[:, 1]
Outputs[:, 2] = (K0[2] + Ks[2]*abs(Inputs[:, 2])**3)*Inputs[:, 2]

np.random.seed()
Test_inp = 1 - 2*np.random.random((num_nodes, num_states))

Test_out[:, 0] = (K0[0] + Ks[0]*abs(Test_inp[:, 0])**3)*Test_inp[:, 0]
Test_out[:, 1] = (K0[1] + Ks[1]*abs(Test_inp[:, 1])**3)*Test_inp[:, 1]
Test_out[:, 2] = (K0[2] + Ks[2]*abs(Test_inp[:, 2])**3)*Test_inp[:, 2]

trace1 = go.Scatter(
    x = Inputs[:, 0],
    y = Outputs[:, 0]/((abs(Outputs).max()).max()),
    mode = 'markers'
)

trace2 = go.Scatter(
    x = Inputs[:, 1],
    y = Outputs[:, 1]/((abs(Outputs).max()).max()),
    mode = 'markers'
)

trace3 = go.Scatter(
    x = Inputs[:, 2],
    y = Outputs[:, 2]/((abs(Outputs).max()).max()),
    mode = 'markers'
)

#data = [trace1, trace2, trace3]
#fig = go.Figure(data=data)

#py.iplot(fig, filename='torques')

#from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.layers import Dense, Activation
import keras
from keras.callbacks import EarlyStopping

# Initialize the constructor
model = Sequential()
act = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)

# Add an input layer 
model.add(Dense(8, input_shape=(3,)))
model.add(Activation(act))

# Add hidden layer
model.add(Dense(8))
model.add(Activation(act))

## Add hidden layer
#model.add(Dense(8))
#model.add(Activation(act))

# Add hidden layer
model.add(Dense(8))
model.add(Activation(act))

# Add Output layer 
model.add(Dense(3, activation='linear'))

model.compile(loss = 'mean_squared_error', optimizer = 'sgd')

StopCond = [EarlyStopping(monitor='loss', patience=3, verbose=0),]


model.fit(Inputs, Outputs/((abs(Outputs).max()).max()),
          epochs = 100000, batch_size = 1000, callbacks = StopCond)


predictions = model.predict(Test_inp)

trace4 = go.Scatter(
    x = Test_inp[:, 0],
    y = predictions[:, 0],
    mode = 'markers'
)

trace5 = go.Scatter(
    x = Test_inp[:, 1],
    y = predictions[:, 1],
    mode = 'markers'
)

trace6 = go.Scatter(
    x = Test_inp[:, 2],
    y = predictions[:, 2],
    mode = 'markers'
)


data2 = [trace1, trace2, trace3, trace4, trace5, trace6]
fig2 = go.Figure(data=data2)

py.iplot(fig2, filename='fit_test')


