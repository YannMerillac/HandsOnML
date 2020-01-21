from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import PIL.Image as Image
import numpy as np
import os

IMG_SIZE = 128
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# step 1
# download dataset at: https://www.robots.ox.ac.uk/~vgg/data/pets/
dataset_dir = "C:/Users/ymerillac/Workspace/tensorflow_datasets/oxford_iiit_pet/3.0.0/images"
mask_dir = "C:/Users/ymerillac/Workspace/tensorflow_datasets/oxford_iiit_pet/3.0.0/annotations/trimaps"
print("Exploring : ",dataset_dir)
list_files = []
list_labels = []
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        if not filename.startswith('.'):
            list_files.append(os.path.join(dataset_dir,filename))
            label_file = filename.replace('.jpg','.png')
            list_labels.append(os.path.join(mask_dir,label_file))
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
    label_string = tf.io.read_file(label)
    label_decoded = tf.image.decode_png(label_string)
    label_resized = tf.image.resize(label_decoded, IMAGE_SHAPE)
    label = tf.cast(label_resized, tf.float32)-1.
    return image, label

dataset = dataset.map(_parse_function)

for image, label in dataset.take(10):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy().shape)
    
    
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

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in dataset.take(1):
    sample_image, sample_mask = image, mask
    
display([sample_image, sample_mask])


dataset = prepare_for_training(dataset, BATCH_SIZE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.
    
    Conv2DTranspose => Batchnorm => Dropout => Relu
    
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    
    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result

up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        
show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 2
STEPS_PER_EPOCH=230    
model_history = model.fit(dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          callbacks=[DisplayCallback()])
  
model.save('PetsSegmentation5Epochs.h5') 