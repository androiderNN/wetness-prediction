from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent.parent.parent

DOWNLOADS_DIR = PARENT_DIR / "downloads"
TRAIN_CSV = DOWNLOADS_DIR / "train.csv"
TEST_CSV = DOWNLOADS_DIR / "test.csv"

