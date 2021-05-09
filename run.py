from bean.Container import Container
from pipeline.prepare_dataset import PrepareDataset
from pipeline.preprocess_captions import PreprocessCaptions
from pipeline.preprocess_images import PreprocessImages


def run():
    container = Container()
    container = PrepareDataset(container).execute()
    container = PreprocessImages(container).execute()
    container = PreprocessCaptions(container).execute()


if __name__ == '__main__':
    run()
