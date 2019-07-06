import keras
import numpy as np
import pandas as pd

from keras.models import model_from_json
#Loading the saving model from file
json_file = open("ANN.json", "r")
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("ANN.h5")

#Compiling before usage
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

y_pred = classifier.predict(X_test)
print(y_pred)
print(y_test)
'''
for i in y_pred[1]:
    for j in y_pred[1][i]:
        if (y_pred[1][i] > 0.5 and y_pred[1][i] <= 1):
            y_pred[i] = 1
        elif (y_pred[1][i] < 0.5 and y_pred[1][i] >= 0):
            y_pred[1][i] = 0
        else:
            raise Exception('sth wrong')
'''
y_pred = (y_pred > 0.5)
print(y_pred)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

