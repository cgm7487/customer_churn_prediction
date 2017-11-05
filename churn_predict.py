
# coding: utf-8

# In[70]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn import preprocessing
import matplotlib.pyplot as plt

def show_train_history(trainHistory, train, validation):
    plt.plot(trainHistory.history[train])
    plt.plot(trainHistory.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def load_data_from_csv(filename):
    csvReader = pd.read_csv(filename)
    print(type(csvReader))
    return csvReader

def preprocess_data(customerData):
    customerData['Churn?'] = customerData['Churn?'].astype('category').cat.codes
    customerData['State'] = customerData['State'].astype('category').cat.codes
    customerData["Int'l Plan"] = customerData["Int'l Plan"].astype('category').cat.codes
    customerData['VMail Plan'] = customerData['VMail Plan'].astype('category').cat.codes
    customerData['Phone'] = customerData['Phone'].astype('category').cat.codes
    #print(customerData.head())
    #print(customerData.info)
    ndarray = customerData.values
    print(ndarray.shape)
    features = ndarray[:, 0:-1]
    label = ndarray[:, -1]
    cateLabel = np_utils.to_categorical(label)
    min_max_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = min_max_scale.fit_transform(features)
    
    return scaledFeatures, cateLabel


customerData = load_data_from_csv("churn.txt")
mask = np.random.rand(len(customerData)) < 0.8
trainDf = customerData[mask]
testDf = customerData[~mask]
trainingFeatures, trainingLabel = preprocess_data(trainDf)
testFeatures, testLabel = preprocess_data(testDf)

#print('total:', len(customerData),
#      'train:', len(trainDf),
#      'test:', len(testDf))

model = Sequential()
model.add(Dense(units=80, input_dim=len(customerData.columns)-1, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=40, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=20, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
trainHistory = model.fit(x=trainingFeatures, y=trainingLabel, validation_split=0.1, epochs=300, batch_size=100, verbose=2)
show_train_history(trainHistory, 'acc', 'val_acc')

scores = model.evaluate(x=testFeatures, y=testLabel)
print(scores[1])



