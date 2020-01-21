from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np
import os

LABEL_LIST = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
              'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer',
              'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel',
              'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
              'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian',
              'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier',
              'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
              'wheaten_terrier', 'yorkshire_terrier']

IMG_SIZE = 128
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


# step 1
dataset_dir = "C:/Users/ymerillac/Workspace/tensorflow_datasets/oxford_iiit_pet/3.0.0/images"
print("Exploring : ",dataset_dir)
list_files = []
list_labels = []
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        if not filename.startswith('.'):
            list_files.append(os.path.join(dataset_dir,filename))
            label_id = "_".join(filename.split('_')[:-1])
            label = np.zeros(37)
            label[LABEL_LIST.index(label_id)] = 1.
            list_labels.append(label)
print(" - Found "+str(len(list_files))+" images ...")
print(" - Found "+str(len(list_labels))+" labels ...")

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(list_labels)))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, IMAGE_SHAPE)
    image = tf.cast(image_resized, tf.float32)/255.
    return image, label

dataset = dataset.map(_parse_function)

for image, label in dataset.take(10):
    print(label)
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    
    
def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

#dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)  
#dataset = dataset.batch(BATCH_SIZE)
dataset = prepare_for_training(dataset, BATCH_SIZE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        label_id = LABEL_LIST[np.argmax(label_batch[n])]
        plt.title(label_id)
        plt.axis('off')
    plt.show()

      
image_batch, label_batch = next(iter(dataset))


#show_batch(image_batch.numpy(), label_batch.numpy())
# dataset = dataset.batch(10)
# 
# # step 4: create iterator and final input tensor
# iterator = dataset.make_one_shot_iterator()
# images, labels = iterator.get_next()

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(37,activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(),   #tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(dataset,
                    steps_per_epoch=230,
                    epochs=10)

model.save('PetsClassification.h5') 
