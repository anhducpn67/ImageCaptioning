import pickle

import tensorflow as tf

import project_config as config


class PreprocessCaptions:
    def __init__(self, container):
        self.container = container
        self.captions_features_vector = None
        self.tokenizer = None
        self.max_caption_length = None

    @staticmethod
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    def tokenizer_captions(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.VOCAB_SIZE,
                                                               oov_token="<unk>",
                                                               filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(self.container.RAW_CAPTIONS)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def create_tokenized_vectors(self):
        self.captions_features_vector = self.tokenizer.texts_to_sequences(self.container.RAW_CAPTIONS)
        self.max_caption_length = self.calc_max_length(self.captions_features_vector)
        self.captions_features_vector = tf.keras.preprocessing.sequence.pad_sequences(self.captions_features_vector,
                                                                                      padding='post')

    def push_data_to_container(self):
        self.container.MAX_CAPTIONS_LENGTH = self.max_caption_length
        self.container.CAPTIONS_FEATURES_VECTOR = self.captions_features_vector
        self.container.TOKENIZER = self.tokenizer

    def cache_data(self):
        pickle.dump(self.captions_features_vector, open(config.CAPTIONS_FEATURES_VECTOR_PATH, 'wb'))
        pickle.dump(self.tokenizer, open(config.TOKENIZER_PATH, 'wb'))

    def load_data_from_cache(self):
        self.container.CAPTIONS_FEATURES_VECTOR = pickle.load(open(config.CAPTIONS_FEATURES_VECTOR_PATH, 'rb'))
        self.container.TOKENIZER = pickle.load(open(config.TOKENIZER_PATH, 'rb'))
        self.container.MAX_CAPTIONS_LENGTH = self.calc_max_length(self.container.CAPTIONS_FEATURES_VECTOR)

    def execute(self):
        print("Preprocessing captions...")
        if config.IS_CACHE_PREPROCESSED_DATA:
            self.load_data_from_cache()
        else:
            self.tokenizer_captions()
            self.create_tokenized_vectors()
            self.push_data_to_container()
            self.cache_data()
        print("Done!")
        return self.container
