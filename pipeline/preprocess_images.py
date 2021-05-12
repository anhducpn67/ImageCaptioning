import pickle

import tensorflow as tf
from tqdm import tqdm

import project_config as config


class PreprocessImages:
    def __init__(self, container):
        self.container = container
        self.image_features_extract_model = None
        self.image_dataset = None
        self.image_path_to_features_vector = dict()
        self.image_features_vector = list()

    @staticmethod
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  # Three channels includes (Red, Green, Blue)
        img = tf.image.resize(img, (config.INCEPTION_V3_IMG_SIZE, config.INCEPTION_V3_IMG_SIZE))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def load_pretrained_inception_v3_model(self):
        InceptionV3_model = tf.keras.applications.InceptionV3(include_top=False,
                                                              weights='imagenet')
        new_input = InceptionV3_model.input
        hidden_layer = InceptionV3_model.layers[-1].output
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    def features_extracted_img(self):
        # Get unique images
        encode_train = sorted(set(self.container.RAW_IMAGES_PATH))

        # Feel free to change batch_size according to your system configuration
        self.image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        self.image_dataset = self.image_dataset.map(
            self.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

    def get_path_to_features(self):
        for img, path in tqdm(self.image_dataset):
            batch_features = self.image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                features = bf.numpy()
                self.image_path_to_features_vector[path_of_feature] = features

    def get_image_features_vector(self):
        for path in self.container.RAW_IMAGES_PATH:
            self.image_features_vector.append(self.image_path_to_features_vector[path])

    def push_data_to_container(self):
        self.container.IMAGES_FEATURES_VECTOR = self.image_features_vector

    def cache_data(self):
        pickle.dump(self.image_features_vector, open(config.IMAGE_FEATURES_VECTOR_PATH, 'wb'))

    def load_data_from_cache(self):
        self.container.IMAGES_FEATURES_VECTOR = pickle.load(open(config.IMAGE_FEATURES_VECTOR_PATH, 'rb'))

    def execute(self):
        print("Preprocessing images...")
        if config.IS_CACHE_PREPROCESSED_DATA:
            self.load_pretrained_inception_v3_model()
            self.load_data_from_cache()
        else:
            self.load_pretrained_inception_v3_model()
            self.features_extracted_img()
            self.get_path_to_features()
            self.get_image_features_vector()
            self.push_data_to_container()
            self.cache_data()
        print("Done!")
        return self.container
