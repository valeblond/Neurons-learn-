#CNN


#PART 1 - data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#we want to create a np array(not a simple vector); that's why not just 1.
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping (3 dimensions)(before fitting or predicting we should input such format)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#for first running(for fitting and writing wages to files
'''
#PART 2 - building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising,creating and training the RNN
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#Saving our training model
#generate the description of the model in json format
model_json = regressor.to_json()

#write the model to the file
json_file = open("RNN.json", "w")
json_file.write(model_json)
json_file.close()

#write weights to the file
regressor.save_weights("RNN.h5")
'''

#LOAD our network from files(weights)
from keras.models import model_from_json
json_file = open("RNN.json", "r")
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
regressor.load_weights("RNN.h5")

regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#PART 3 - making the predictions and visualising the results
#Getting the real stock price of the month that we want to predict
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
#we should predict 20 days(that's the reason why 80)
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

