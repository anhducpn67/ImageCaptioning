import collections
import random

import tensorflow as tf

import project_config as config


class CreateDataset:
    def __init__(self, container):
        self.container = container
        self.img_to_cap_vector = collections.defaultdict(list)
        self.img_to_features = {}
        self.val_img = []
        self.val_cap = []
        self.val_img_path = []
        self.train_img = []
        self.train_cap = []
        self.train_dataset = None

    def get_img_to_cap_vector(self):
        for img, cap in zip(self.container.RAW_IMAGES_PATH, self.container.CAPTIONS_FEATURES_VECTOR):
            self.img_to_cap_vector[img].append(cap)

    def get_img_to_features(self):
        for img, features in zip(self.container.RAW_IMAGES_PATH, self.container.IMAGES_FEATURES_VECTOR):
            self.img_to_features[img] = features

    def split_data(self):
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        slice_index = int(len(img_keys) * config.PERCENTAGE_SPLIT_DATA)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]
        for img in img_name_train_keys:
            capt_len = len(self.img_to_cap_vector[img])
            self.train_img.extend([self.img_to_features[img]] * capt_len)
            self.train_cap.extend(self.img_to_cap_vector[img])

        for img in img_name_val_keys:
            capv_len = len(self.img_to_cap_vector[img])
            self.val_img.extend([self.img_to_features[img]] * capv_len)
            self.val_cap.extend(self.img_to_cap_vector[img])
            self.val_img_path.extend([img] * capv_len)

    def create_dataset(self):
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_img, self.train_cap))
        # Shuffle and batch
        self.train_dataset = self.train_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def push_data_to_container(self):
        self.container.TRAIN_DATASET = self.train_dataset
        self.container.NUM_STEPS = len(self.train_img)
        self.container.VAL_IMAGES = self.val_img
        self.container.VAL_CAPTIONS = self.val_cap
        self.container.VAL_IMAGES_PATH = self.val_img_path

    def execute(self):
        print("Creating dataset...")
        self.get_img_to_cap_vector()
        self.get_img_to_features()
        self.split_data()
        self.create_dataset()
        self.push_data_to_container()
        print("Done!")
        return self.container
