#TESTING THE MODEL(PUT SOME IMAGES FOR RECOGNITION)


#LOAD our network from files(weights)
from keras.models import model_from_json
json_file = open("CNN.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("CNN.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#IMAGE augmentation(image preprocessing)
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


#PREDICTION of two different images
def choose(prediction):
    if prediction == 1:
        pred = 'dog'
        print(pred)
    elif prediction == 0:
        pred = 'cat'
        print(pred)
    else:
        raise ValueError("Wrong class")

import numpy as np
from keras.preprocessing import image

#1st image
img = image.load_img('dataset\single_prediction\cat_or_dog_1.jpg', target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
prediction = loaded_model.predict(img)
print(training_set.class_indices)
choose(prediction[0][0])

#2nd image
img = image.load_img('dataset\single_prediction\cat_or_dog_2.jpg', target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
prediction = loaded_model.predict(img)
print(training_set.class_indices)
choose(prediction[0][0])


