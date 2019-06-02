#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:35:46 2019

@author: riz04
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

# selecting the right column we are interested in
# creating a numpy array, taking range 1 to 2 and adding .values will do that
training_set = dataset_train.iloc[:,1:2].values

# feature scaling
# two best practices are standardisation & normalisation
# when we have sigmoid fn as activation fn in o/p layer of RNN
# it is suggested to apply RNN
from sklearn.preprocessing import MinMaxScaler

# feature range - (0,1), bcz we want all prices to be in this range
SC = MinMaxScaler(feature_range=(0,1))
training_set_scaled = SC.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
# 60 timesteps means that at each time T RNN is going to look at
# 60 stock price bfore time T
# and based on the trends its capturing during 60 previous timesteps
# it will try to predict next output

# X_train = for each financial day, it will contain 60 previous stock prices
# Y_train = it contains stock price next financial day
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)


# reshaping
# dimensons (batch_size, timesteps, number of indicators - google stock price here)
X_train = np.reshape(X_train,(X_train.shape[0] , X_train.shape[1], 1))

# building the RNN
# sequential class allows to create neural network object representing 
# sequence of layers
# dense class to define output layer
# LSTM class to add LSTM layers]
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# intialising the RNN
# here we are predicting continous stock price, so we are doing regression
regressor = Sequential()

# adding the first LSTM layer and some dropout regularization
# arguments of LSTM layer are
# units - number of LSTM cells or memory units in this LSTM layer
# return_sequences - true (bcz we are building stacked LSTM, we will 
# have several layers)
# input shape = shape of X train, we will only specify two dimesnions here
# bcz the first dimension will be considered automatically
regressor.add(LSTM(units = 50, return_sequences=True, 
                   input_shape = (X_train.shape[1], 1)))

# 20% of the neurons will be ignored during training(forward and backward 
# propagation)
regressor.add(Dropout(0.2))

# adding second LSTM Layer
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding third LSTM Layer
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding fourth layer
regressor.add(LSTM(units = 50, return_sequences=False))
regressor.add(Dropout(0.2))

# adding final or output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer= "adam", loss = "mean_squared_error" )

# fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# making the predictions and visualising the results

# getting the real stock price of Jan 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price  = dataset_test.iloc[:,1:2].values

# getting the predicted stock price of Jan 2017
# - since we trained our model on sacled inputs
# - we need to scale both train and test together to maintain
# - consistency and predict better results

dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

# for input
# lower bound = first financial day of jan 2017 we are predicting - 60
# upper bound = last financial day of jan 2017 we are predicting - 60

# 3rd jan 2017 - 60
# we add .values to convert this df into a numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values

# since we didn't use iloc method, we update inputs to get correct numpy shape
# i.e in lines and in colours
inputs = inputs.reshape(-1,1)

# we must directly apply transform and not fit_transform'
# bcz sc was fitted previously on training data]
# and we need the same scale here
inputs = SC.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)


# reshaping
# dimensons (batch_size, timesteps, number of indicators - google stock price here)
X_test = np.reshape(X_test,(X_test.shape[0] , X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = SC.inverse_transform(predicted_stock_price)

# visualising the results
plt.plot(real_stock_price, color = "red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Stock Price")

plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()


￼
￼
#results

# in the parts of predictions, which contains some spikes
# our predictions lags behind actual values, bcz our model couldn't react 
# to fast non-linear changes
# on the other hand, for the part of predictions containing 
# smooth changes, our model reacts pretty well
# and manages to follow
# the upward and downward trends










































    
    






















