import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

    def build_graph(self):
        input_layer = tf.keras.layers.Input(shape=(64, 2048))
        return tf.keras.Model(inputs=[input_layer], outputs=self.call(input_layer))
