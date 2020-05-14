import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt

#So, let's load the MNIST dataset from the Keras library. The dataset is readily divided into a training set and a test set.
# import the data
from keras.datasets import mnist

# read the data-------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Let's confirm the number of images in each set. According to the dataset's documentation, 
#we should have 60000 images in X_train and 10000 images in the X_test.
X_train.shape

#Let's visualize the first image in the training set using Matplotlib's scripting layer.
plt.imshow(X_train[0])

#With conventional neural networks, we cannot feed in the image as input as is. -----------------------------------------
#So we need to flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784.
# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

#Since pixel values can range from 0 to 255, let's normalize the vectors to be between 0 and 1----------------------
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#Finally, before we start building our model, remember that for classification we need to divide our target variable into categories. ----
#We use the to_categorical function from the Keras Utilities package.
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

#Build a Neural Network---------------------------------------------
# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Train and Test the Network
# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)


#Let's print the accuracy and the corresponding error.
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))     

model.save('classification_model.h5')

#Since our model contains multidimensional arrays of data, then models are usually saved as .h5 files.
#When you are ready to use your model again, you use the load_model function from keras.models.
from keras.models import load_model

pretrained_model = load_model('classification_model.h5')