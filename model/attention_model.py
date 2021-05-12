from model.cnn_encoder_layer import CNN_Encoder
from model.rnn_decoder_layer import RNN_Decoder


class AttentionModel:
    def __init__(self, embedding_dim, units, vocab_size):
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
