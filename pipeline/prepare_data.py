import collections
import json
import pickle
import random

import project_config as config


class PrepareDataset:
    def __init__(self, container):
        self.raw_images_path = []
        self.raw_captions = []
        self.container = container
        self.image_path_to_caption = collections.defaultdict(list)
        self.train_image_paths = list()

    def get_image_path_to_caption(self):
        annotations = json.load(open(config.ANNOTATIONS_FILE_PATH, 'r'))
        for val in annotations['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_paths = config.IMAGES_FOLDER_PATH + '/COCO_train2014_' + '%012d.jpg' % (val['image_id'])
            self.image_path_to_caption[image_paths].append(caption)

    def get_train_image_paths(self):
        self.train_image_paths = list(self.image_path_to_caption.keys())
        random.shuffle(self.train_image_paths)

    def limit_dataset(self):
        self.train_image_paths = self.train_image_paths[:config.SIZE_LIMIT_IMG_DATASET]

    def get_train_data(self):
        for image_path in self.train_image_paths:
            caption_list = self.image_path_to_caption[image_path]
            self.raw_captions.extend(caption_list)
            self.raw_images_path.extend([image_path] * len(caption_list))

    def push_data_to_container(self):
        self.container.RAW_CAPTIONS = self.raw_captions
        self.container.RAW_IMAGES_PATH = self.raw_images_path

    def cache_data(self):
        pickle.dump(self.raw_captions, open(config.RAW_CAPTIONS_PATH, 'wb'))
        pickle.dump(self.raw_images_path, open(config.RAW_IMAGES_PATH, 'wb'))

    def load_data_from_cache(self):
        self.container.RAW_CAPTIONS = pickle.load(open(config.RAW_CAPTIONS_PATH, 'rb'))
        self.container.RAW_IMAGES_PATH = pickle.load(open(config.RAW_IMAGES_PATH, 'rb'))

    def execute(self):
        print("Preparing data...")
        if config.IS_CACHE_PREPROCESSED_DATA:
            self.load_data_from_cache()
        else:
            self.get_image_path_to_caption()
            self.get_train_image_paths()
            if config.SIZE_LIMIT_IMG_DATASET is not None:
                self.limit_dataset()
            self.get_train_data()
            self.push_data_to_container()
            self.cache_data()
        print("Done!")
        return self.container
