#SOM+ANN

#PART 1- FIND FRAUDS(SOM)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#Training the SOM
#using for this file that was created by some developer
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

#Finding the frauds
mappings = som.win_map(X)
frauds = mappings[(7, 5)]
# np.concatenate((mappings[(7, 5)], mappings[(6, 6)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds)


#PART 2 - UNSUPERVISED TO SUPERVISED(ANN)
#Creating matrix of features
customers = dataset.iloc[:, 1:].values

#Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialisation and creating the ANN
classifier = Sequential()
classifier.add(Dense(2, init='uniform', activation='relu', input_dim=15))
classifier.add(Dense(1, init='uniform', activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(customers, is_fraud, nb_epoch=2, batch_size=1)
