from typing import Any


class Container:
    """
    Used to carrying object through out the pipeline
    """

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __init__(self):
        self.TRAIN_CAPTIONS = None
        self.TRAIN_IMG_PATH = None
