from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent.parent.parent

DOWNLOADS_DIR = PARENT_DIR / "downloads"
TRAIN_CSV = DOWNLOADS_DIR / "train.csv"
TEST_CSV = DOWNLOADS_DIR / "test.csv"

DATA_DIR = PARENT_DIR / "data"
TRAIN_IMAGE_DIR = DATA_DIR / "image_train"
TEST_IMAGE_DIR = DATA_DIR / "image_test"

OUTPUT_DIR = PARENT_DIR / "output"

