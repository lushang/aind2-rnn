import numpy as np
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# DONE: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for idx in range(len(series) - window_size):
        X.append(series[idx : idx + window_size])
        y.append(series[window_size + idx])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
# •layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
# •layer 2 uses a fully connected module with one unit
def build_part1_RNN(window_size):
    model = Sequential()
    #model.add(Dense(5, input_shape = (window_size,1)))
    model.add(LSTM(5, activation='tanh', recurrent_activation='hard_sigmoid', input_shape = (window_size,1)))
    model.add(Dense(1))
    return model

### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    punctuation += list(string.ascii_lowercase)
    punctuation += [' ']
    new_text = ''
    for c in text:
        if c in punctuation:
            new_text += c
    text = new_text
    return text

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for idx in range(0, len(text) - window_size, step_size):
        inputs.append(text[idx : idx + window_size])
        outputs.append(text[window_size + idx])

    return inputs,outputs

# DONE build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model