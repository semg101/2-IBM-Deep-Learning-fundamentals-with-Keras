#1. Download and Clean Dataset 2. Import Keras 3. Build a Neural Network 4. Train and Test the Network

#Download and Clean Dataset
#Let's start by importing the pandas and the Numpy libraries.
import pandas as pd
import numpy as np

#Let's download the data and read it into a pandas dataframe.---------------------------------------------------
concrete_data = pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
concrete_data.head()

#Let's check how many data points we have
concrete_data.shape

#So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.
#Let's check the dataset for any missing values.
concrete_data.describe()

concrete_data.isnull().sum()

#Split data into predictors and target-----------------------------------------------------------------------------
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#Let's do a quick sanity check of the predictors and the target dataframes
predictors.head()

target.head()

#Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation-------------
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors

#Let's go ahead and import the Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#Build a Neural Network---------------------------------------------------------------------------------------------------------
#Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


#Train and Test the Network--------------------------------------------------------------------
# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
