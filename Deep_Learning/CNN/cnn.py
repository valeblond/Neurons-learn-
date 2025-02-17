#CNN


#PART 1 - Building the CNN
#Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# from tensorflow.python.keras import backend

#Initialisation and creating the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Image augmentation(for having more different images(learning different characteristics all the time))
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary'
)

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8000,
                        epochs=10,
                        validation_data=test_set,
                        validation_steps=2000
)

#Saving our training model
#generate the description of the model in json format
model_json = classifier.to_json()

#write the model to the file
json_file = open("CNN.json", "w")
json_file.write(model_json)
json_file.close()

#write weights to the file
classifier.save_weights("CNN.h5")









