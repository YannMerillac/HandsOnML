# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np

LABEL_LIST = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

IMG_SIZE = 32


def predict_img(input_file,model,class_names):
    test_img = Image.open(input_file).resize((IMG_SIZE,IMG_SIZE))
    test_vect = np.array(test_img)/255.
    prediction = model.predict(test_vect[np.newaxis, ...])
    print(prediction)
    print(np.max(prediction))
    predicted_label = class_names[np.argmax(prediction[0])]
    plt.imshow(test_img)
    plt.axis('off')
    _ = plt.title("Prediction: " + predicted_label)
    plt.show()


# load model
#model = tf.keras.models.load_model('PetsClassification.h5')
model = tf.keras.models.load_model('Cifrar10CnnModel10Epochs.h5')

# Show the model architecture
model.summary()


#predict_img("camion.jfif",model,LABEL_LIST)
predict_img("petrolier.jpeg",model,LABEL_LIST)
