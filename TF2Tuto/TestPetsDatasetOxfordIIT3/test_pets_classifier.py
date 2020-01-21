from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np

LABEL_LIST = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
              'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer',
              'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel',
              'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
              'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian',
              'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier',
              'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
              'wheaten_terrier', 'yorkshire_terrier']

IMG_SIZE = 128


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
model = tf.keras.models.load_model('PetsClassification20Epochs.h5')

# Show the model architecture
model.summary()


predict_img("TestImg/test_pets5.jpg",model,LABEL_LIST)