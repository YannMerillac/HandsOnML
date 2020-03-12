# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image

import tensorflow as tf

from tensorflow.keras import layers, models


def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv)
    df.fillna(method = 'ffill',inplace = True)
    print(df.info())
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
    X = np.asarray([df['Image']], dtype=np.float32).reshape(df.shape[0],96,96,1)/255.
    y = df.drop(['Image'], axis=1).to_numpy()
    return X,y

X,y = load_data_from_csv('facial-keypoints-detection/training/training.csv')
print(X.shape, y.shape)
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plt.imshow(X[i,:,:].reshape((96,96)),cmap='gray')
    #get_image_and_dots(training, i)
    for j in range(0,30,2):
        plt.plot(y[i,j], y[i,j+1], 'ro')

plt.show()

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

#model = models.Sequential()
#model.add(layers.Conv2D(32, (11, 11), activation=lrelu, use_bias=False, input_shape=(96, 96, 1)))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
##model.add(layers.Dropout(0.2))
##
#model.add(layers.Conv2D(64, (5, 5), activation=lrelu, use_bias=False))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
##model.add(layers.Dropout(0.3))
##
#model.add(layers.Conv2D(128, (3, 3), activation=lrelu, use_bias=False))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
##model.add(layers.Dropout(0.5))
#
##model.add(layers.Conv2D(128, (3, 3), activation='relu', use_bias=False))
##model.add(layers.BatchNormalization())
##model.add(layers.MaxPooling2D((2, 2)))
##
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation=lrelu))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(512, activation=lrelu))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.1))
#model.add(layers.Dense(30))

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Flatten())
##@model.add(layers.Dropout(0.3))
#model.add(layers.Dense(512, activation='relu'))
##model.add(layers.Dropout(0.3))
#model.add(layers.Dense(30))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.2))
#
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.3))
#
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.5))
#
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
#model.add(layers.Dense(512, activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(30))
model.summary()


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

def plot_face_keypoints(model,img_vect):
    pt_predicts = model.predict(img_vect[np.newaxis, ...])
    plt.imshow(img_vect.reshape((96,96)),cmap='gray')
    for j in range(0,30,2):
        plt.plot(pt_predicts[0,j], pt_predicts[0,j+1], 'ro')
    plt.show()


def load_img_from_file(img_filename):
    img = Image.open(img_filename).convert('L').resize((96,96))
    test_vect = np.array(img)/255.
    #print(test_vect)
    #print(test_vect.shape)
    #plt.imshow(test_vect.reshape((96,96)),cmap='gray')
    #plt.show()
    return test_vect.reshape((96,96,1))

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #clear_output(wait=True)
        test_img = load_img_from_file("selfie1.jpg")
        plot_face_keypoints(model,test_img)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


N_EPOCHS = 100
history = model.fit(X, y, epochs=N_EPOCHS,
                    validation_split = 0.2,
                    callbacks=[DisplayCallback()])

model.save('SimpleCnn'+str(N_EPOCHS)+'Epochs_V2.h5')

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.ylim([0., 5.])
plt.legend(loc='lower right')
plt.show()

