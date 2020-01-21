from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np

IMG_SIZE = 128
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE)

def show_predictions(model,input_file):
    test_img = Image.open(input_file).resize((IMG_SIZE,IMG_SIZE))
    test_vect = np.array(test_img)/255.
    # plot image
    plt.subplot(1, 2, 1)
    plt.title("Test Image")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(test_vect))
    plt.axis('off')
    # plot mask
    plt.subplot(1, 2, 2)
    plt.title("Predicted mask")
    prediction = model.predict(test_vect[np.newaxis, ...])[0]
    prediction = tf.argmax(prediction, axis=-1)
    prediction = prediction[..., tf.newaxis]
    plt.imshow(tf.keras.preprocessing.image.array_to_img(prediction))
    plt.axis('off')
    plt.show()
    
    
# load model
#model = tf.keras.models.load_model('PetsClassification.h5')
model = tf.keras.models.load_model('PetsSegmentation5Epochs.h5')

# Show the model architecture
model.summary()


show_predictions(model,"TestImg/Pets.jpg")

