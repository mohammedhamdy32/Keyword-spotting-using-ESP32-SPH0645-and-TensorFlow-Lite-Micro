
class   Config:

    # Constants for audio process during Quantization and Evaluation
    SAMPLE_RATE = 16000
    CLIP_DURATION_MS = 1000
    LENGHT_OF_VOICE = 16000
    WINDOW_SIZE_MS = 30
    STRIDE_SIZE_MS = 20
    FEATURE_BIN_COUNT = 40
    BACKGROUND_FREQUENCY = 0.8
    BACKGROUND_VOLUME_RANGE = 0.1
    TIME_SHIFT_MS = 100.0

    SPECTOGRAM_ROW = 49
    SPECTOGRAM_COL = 40

    START_LEARNING_RATE = 0.001
    LEARNING_RATE_EPOCH_CHANGE = 10  # Means the epoch which the learning rate will decreses exponentially

    # Noise floor to detect if any audio is present
    NOISE_FLOOR=0.1

    NUMBER_OF_CLASSES = 4

    TRAIN_RATIO = 0.7
    TEST_RATIO  = 0.3

    NOISE_FOLDER_PATH   = 'noise_clips'
    DATASET_FOLDER_PATH = 'dataset'

    # other datasets
    PETER_WARDEN_PATH = '../dataset/Pete_Warden_dataset'
    MY_VOICE_WORDS_PATH = '../dataset/My_voice_dataset'
    ARABIC_WORDS_PATH = '../dataset/archive/dataset/dataset'

    BATCH_SIZE = 8


