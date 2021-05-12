from bean.Container import Container
from pipeline.create_dataset import CreateDataset
from pipeline.evaluate_model import EvaluateModel
from pipeline.prepare_data import PrepareDataset
from pipeline.preprocess_captions import PreprocessCaptions
from pipeline.preprocess_images import PreprocessImages
from pipeline.train_model import TrainModel


def run():
    container = Container()
    container = PrepareDataset(container).execute()
    container = PreprocessImages(container).execute()
    container = PreprocessCaptions(container).execute()
    container = CreateDataset(container).execute()
    container = TrainModel(container).execute()
    container = EvaluateModel(container).execute()
    # from model.bahdanau_attention_layer import BahdanauAttention
    # from model.cnn_encoder_layer import CNN_Encoder
    # from model.rnn_decoder_layer import RNN_Decoder
    # from tensorflow import keras
    # model = BahdanauAttention(project_config.UNITS)
    # keras.utils.plot_model(model.build_graph(), "bahdanau_attention.png", show_shapes=True)
    # model = CNN_Encoder(project_config.EMBEDDING_DIM)
    # keras.utils.plot_model(model.build_graph(), "cnn_encoder.png", show_shapes=True)
    # model = RNN_Decoder(project_config.EMBEDDING_DIM, project_config.UNITS, project_config.VOCAB_SIZE)
    # keras.utils.plot_model(model.build_graph(), "rnn_decoder.png", show_shapes=True)


if __name__ == '__main__':
    run()
