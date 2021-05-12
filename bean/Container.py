from typing import Any


class Container:
    """
    Used to carrying object through out the pipeline
    """

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __init__(self):
        self.RAW_CAPTIONS = None
        self.RAW_IMAGES_PATH = None

        self.CAPTIONS_FEATURES_VECTOR = None
        self.IMAGES_FEATURES_VECTOR = None

        self.MAX_CAPTIONS_LENGTH = None
        self.TOKENIZER = None
        self.NUM_STEPS = None

        self.TRAIN_DATASET = None
        self.VAL_IMAGES = None
        self.VAL_CAPTIONS = None
        self.VAL_IMAGES_PATH = None
        self.ATTENTION_MODEL = None
