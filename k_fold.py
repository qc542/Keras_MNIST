from keras.datasets import mnist
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.axes as axes
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from collections import Counter

(x_set, y_set), (x_test, y_test) = mnist.load_data()
x_set = x_set/255

def k_fold_split(x_set, y_set, folds=1):
    """ Inputs: The x_set data from mnist, the y_set labels from mnist.
    Expected Output: the shuffled and K-split datasets."""

    length_fold_x = len(x_set) // folds
    new_x_set = np.zeros((folds, length_fold_x, len(x_set[0]),
        len(x_set[0][0])))

    length_fold_y = len(y_set) // folds
    new_y_set = np.zeros((folds, length_fold_y))
    range_start_x = 0
    range_start_y = 0

    for n in range(folds):
        new_x_set[n] = new_x_set[n] + x_set[range_start_x:range_start_x+length_fold_x]
        range_start_x += length_fold_x

    for n in range(folds):
        new_y_set[n] = new_y_set[n] + y_set[range_start_y:range_start_y+length_fold_y]
        range_start_y += length_fold_y

    return (new_x_set, new_y_set)

x_folds, y_folds = k_fold_split(x_set, y_set, 5)


def construct_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

new_model = construct_model()
flattened_x_set = np.zeros((len(x_folds), len(x_folds[0]), len(x_folds[0][0])**2))

for n in range(len(x_folds)):
    for m in range(len(x_folds[n])):
        flattened_x_set[n][m] = x_folds[n][m].flatten()

modified_y_set = np.zeros((len(y_folds), len(y_folds[0]), 10))
# Creates a new array of the dimensions that fit one-hot encoding

for n in range(len(modified_y_set)):
    for m in range(len(modified_y_set[0])):
        modified_y_set[n][m] = to_categorical(
                y_folds[n][m], num_classes = 10)
        """ Using the to_categorical function of Keras to 
        convert labels into one-hot encoding, the format 
        Keras requires to fit the model."""

new_train_dataset = (flattened_x_set[0], modified_y_set[0])
new_validation_dataset = (flattened_x_set[1], modified_y_set[1])

def train_model(model, train_dataset, validation_dataset, epochs, name):
    x_set, y_set = train_dataset
    model.fit(x=x_set, y=y_set, epochs=epochs, batch_size=128, 
            validation_data = validation_dataset)
    model.save(f'./{name}')

# Epochs are the number of times the dataset will be iterated over; a good number is 20.
train_model(new_model, new_train_dataset, new_validation_dataset, 20, 'train_model_1')


def train_validate_k(x_folds, y_folds, num_folds):
    """ Inputs: x_folds, the x folds returned from the k_fold 
    algorithm above; y_folds, the y folds returned from k_fold; 
    num_folds, the number of folds used to make x_folds and 
    y_folds.
    Expected output: none. This function has no explicit output,
    but there must be num_fold models trained and saved to the 
    disk."""

    for n in range(num_folds):
        model_name = 'model_' + str(n)
        train_dataset = (flattened_x_set[n], modified_y_set[n])
        if n == (num_folds-1):
            validation_dataset = (flattened_x_set[0], 
                modified_y_set[0])
        else:
            validation_dataset = (flattened_x_set[n+1],
                    modified_y_set[n+1])
        train_model(construct_model(), train_dataset, 
                validation_dataset, 20, model_name)

    return

train_validate_k(x_folds, y_folds, 5)
