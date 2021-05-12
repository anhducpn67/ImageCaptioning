IS_CACHE_PREPROCESSED_DATA = True
IS_TRAINED_MODEL = False
SIZE_LIMIT_IMG_DATASET = 4000  # Equal None if don't limit dataset
INCEPTION_V3_IMG_SIZE = 299  # Constant
VOCAB_SIZE = 5000  # Limit words in vocabulary
PERCENTAGE_SPLIT_DATA = 0.8  # Split data to train and validation
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 20
ATTENTION_FEATURES_SHAPE = 64

# Path
ANNOTATIONS_FILE_PATH = './resources/annotations/captions_train2014.json'
IMAGES_FOLDER_PATH = './resources/train2014'

RAW_IMAGES_PATH = './resources/cache/raw_image.pkl'
RAW_CAPTIONS_PATH = './resources/cache/raw_captions.pkl'

IMAGE_FEATURES_VECTOR_PATH = './resources/cache/image_features.pkl'
CAPTIONS_FEATURES_VECTOR_PATH = './resources/cache/captions.pkl'
TOKENIZER_PATH = "./resources/cache/tokenizer.pkl"

TRAIN_DATASET_PATH = "./resources/cache/train_dataset.pkl"
VAL_DATASET_PATH = "./resources/cache/val_dataset.pkl"
CHECKPOINT_PATH = "./resources/checkpoints/train"
