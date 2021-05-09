import collections
import json
import random

import project_config as config


class PrepareDataset:
    def __init__(self, container):
        self.train_img_path = []
        self.train_captions = []
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
            self.train_captions.extend(caption_list)
            self.train_img_path.extend([image_path] * len(caption_list))

    def push_data_to_container(self):
        self.container.TRAIN_CAPTIONS = self.train_captions
        self.container.TRAIN_IMG_PATH = self.train_img_path

    def execute(self):
        self.get_image_path_to_caption()
        self.get_train_image_paths()
        if config.SIZE_LIMIT_IMG_DATASET is not None:
            self.limit_dataset()
        self.get_train_data()
        self.push_data_to_container()
        return self.container
