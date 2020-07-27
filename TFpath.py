import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def transform_image_to_tfrecord_image_path(image_path, BATCH_SIZE):
    BATCH_SIZE = BATCH_SIZE
    data_dir = pathlib.Path(image_path)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    IMG_HEIGHT, IMG_WIDTH = 250, 250
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    #print(list_ds)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        # normalize
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=500):
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

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    BATCH_SIZE = BATCH_SIZE
    return prepare_for_training(labeled_ds)
    # image_batch, label_batch = next(iter(train_ds))
    # print(image_batch.shape)
    # print(label_batch.shape)

    '''
    data_dir = pathlib.Path(image_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/ '*/*'))
    print(list_ds)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    print(labeled_ds)
    '''