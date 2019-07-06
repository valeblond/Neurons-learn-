#ANN

# Part 1 - Data Preprocessing
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1, 2])],            # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough')                                # Leave the rest of the columns untouched
X = np.array(onehotencoder.fit_transform(X), dtype=np.float)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialisation and creating the ANN
classifier = Sequential()
classifier.add(Dense(6, init='uniform', activation='relu', input_dim=13))
classifier.add(Dense(6, init='uniform', activation='relu'))
classifier.add(Dense(1, init='uniform', activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(X_train, y_train, nb_epoch=100, batch_size=10)

#Saving our training model
#Generate the description of the model in json format
model_json = classifier.to_json()

#Write the model to the file
json_file = open("ANN.json", "w")
json_file.write(model_json)
json_file.close()

#Write weights to the file
classifier.save_weights("ANN.h5")






#print(X)
#print(X_train)
#print(X_train.shape)
#print(X_test)
#print(y_pred)
#print(cm)