import os
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy.typing as npt
import cv2
from sklearn.model_selection import train_test_split

from wetness_regression.utils.wrpath import TRAIN_CSV, TEST_CSV, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
from wetness_regression.dataset.load_dataset import WetnessSample, load_csv


@dataclass
class WetnessImageSample(WetnessSample):
    """WetnessSampleの属性に加え画像も保持する"""
    image: npt.NDArray | None = None
    """ロードした画像"""

    @classmethod
    def from_wetnesssample(cls, wnsample: WetnessSample, img_dir: Path) -> "WetnessImageSample":
        """WetnessSampleから構築する"""
        img_path = img_dir / f"{wnsample.id}.jpg"

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not exist.")

        wnsample = asdict(wnsample)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"failed to load image: {img_path}")

        return cls(**wnsample, image=image)


def load_image_samples(csv_path: Path, image_dir: Path) -> list[WetnessImageSample]:
    """CSV と画像ディレクトリから画像付きサンプル一覧を構築する。"""
    samples = load_csv(csv_path)
    return [WetnessImageSample.from_wetnesssample(sample, image_dir) for sample in samples]


def load_split_samples(
    valid_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[WetnessImageSample], list[WetnessImageSample], list[WetnessImageSample]]:
    """train/valid/test の画像付きサンプルをまとめて返す。"""
    train_all_samples = load_image_samples(TRAIN_CSV, TRAIN_IMAGE_DIR)

    train_samples, valid_samples = train_test_split(
        train_all_samples,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
    )

    test_samples = load_image_samples(TEST_CSV, TEST_IMAGE_DIR)

    return train_samples, valid_samples, test_samples

