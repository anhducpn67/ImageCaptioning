import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

import project_config as config


class EvaluateModel:
    def __init__(self, container):
        self.container = container
        self.model = self.container.ATTENTION_MODEL

    def evaluate(self, img_tensor_val):
        attention_plot = np.zeros((self.container.MAX_CAPTIONS_LENGTH,
                                   config.ATTENTION_FEATURES_SHAPE))

        hidden = self.model.decoder.reset_state(batch_size=1)

        features = self.model.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.container.TOKENIZER.word_index['<start>']], 0)
        result = []

        for i in range(self.container.MAX_CAPTIONS_LENGTH):
            predictions, hidden, attention_weights = self.model.decoder(dec_input,
                                                                        features,
                                                                        hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.container.TOKENIZER.index_word[predicted_id])

            if self.container.TOKENIZER.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    @staticmethod
    def plot_attention(image, result, attention_plot):
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for i in range(len_result):
            temp_att = np.resize(attention_plot[i], (8, 8))
            grid_size = max(np.ceil(len_result / 2), 2)
            ax = fig.add_subplot(grid_size, grid_size, i + 1)
            ax.set_title(result[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    def captions_val_set(self):
        rid = np.random.randint(0, len(self.container.VAL_IMAGES))
        img_tensor_val = self.container.VAL_IMAGES[rid]
        img_path_val = self.container.VAL_IMAGES_PATH[rid]
        real_caption = ' '.join([self.container.TOKENIZER.index_word[i]
                                 for i in self.container.VAL_CAPTIONS[rid] if i not in [0]])
        result, attention_plot = self.evaluate(img_tensor_val)

        print('Real Caption:', real_caption)
        print('Prediction Caption:', ' '.join(result))
        EvaluateModel.plot_attention(img_path_val, result, attention_plot)

    def execute(self):
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        self.captions_val_set()
        return self.container
