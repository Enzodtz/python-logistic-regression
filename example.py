# This file is an example of usage of the 'logistic_regression_model.py' file.
# This will call the model to train it to predict if there's a cat on the image
# or not
# The dataset is avaible at the folder 'datasets' of this repo, needed to run
# this script

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import random
from PIL import Image
from scipy import ndimage
from logistic_regression_model import *

# Function to load the dataset
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Getting dataset info
m_train = train_set_x_orig.shape[0]
m_test =  test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Transforming images into a flat array, since they were (num_px, num_px, 3)
train_set_x_flatten = train_set_x_orig.reshape(m_train, num_px * num_px * 3).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, num_px * num_px * 3).T

# Normalizing pixel value
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Training the model
model = LogisticRegressionModel()
stats = model.train(train_set_x, train_set_y, test_set_x, test_set_y)

# Model stats
plt.figure()
plt.title('model costs')
plt.plot(model.costs)
plt.show()

plt.figure()
plt.title('model costs')
plt.bar(stats.keys(), [stats[stat] for stat in stats])
plt.show()

# Predicting with other image, notice that it should be pre-processed to fit the model
image = Image.open("images/cat.jpeg")
img_show = image.resize((num_px, num_px), Image.ANTIALIAS)
image = np.asarray(img_show)
image = image.flatten()
image = image/255.
prediction = model.predict(image)

plt.title('Predicted as: ' + ('cat' if prediction else 'non-cat'))
plt.imshow(img_show)
plt.show()
