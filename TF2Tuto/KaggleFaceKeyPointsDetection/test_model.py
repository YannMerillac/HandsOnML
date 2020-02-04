# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np
import pandas as pd


def plot_face_keypoints(model,img_vect):
    pt_predicts = model.predict(img_vect[np.newaxis, ...])
    plt.imshow(img_vect.reshape((96,96)),cmap='gray')
    for i in range(1,31,2):
        plt.plot(pt_predicts[0][i-1], pt_predicts[0][i], 'ro')    
    plt.show()

def load_dataset_from_csv(csv_filename):
    df = pd.read_csv(csv_filename)
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
    return np.asarray([df['Image']], dtype=np.float32).reshape(df.shape[0],96,96,1)/255.

def load_img_from_file(img_filename):
    img = Image.open(img_filename).convert('L').resize((96,96))
    test_vect = np.array(img)/255.
    print(test_vect)
    print(test_vect.shape)
    #plt.imshow(test_vect.reshape((96,96)),cmap='gray')
    #plt.show()
    return test_vect.reshape((96,96,1))
    

# load model
model = tf.keras.models.load_model('SimpleCnn100Epochs_v2.h5')

test_images = load_dataset_from_csv('facial-keypoints-detection/test/test.csv')
print(test_images.shape)


plot_face_keypoints(model,test_images[888])

test_img = load_img_from_file("selfie1.jpg")
plot_face_keypoints(model,test_img)

