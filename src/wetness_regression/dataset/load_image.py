import os
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt
import cv2
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

from wetness_regression.utils.wrpath import TRAIN_CSV, TEST_CSV, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
from wetness_regression.dataset.load_dataset import WetnessSample, load_csv

# 全訓練データから計算した波形のグローバル統計量（標準化用）
WAVEFORM_GLOBAL_MEAN = 0.132986
WAVEFORM_GLOBAL_STD = 0.210632


@dataclass
class WetnessImageSample(WetnessSample):
    """WetnessSampleの属性に加え画像も保持する"""
    image: npt.NDArray | None = None
    """ロードした画像"""

    @classmethod
    def from_wetnesssample(cls, wnsample: WetnessSample, img_dir: Path) -> "WetnessImageSample":
        """WetnessSampleから構築する"""
        img_path = img_dir / f"{wnsample.id}.png"

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not exist.")

        wnsample = asdict(wnsample)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"failed to load image: {img_path}")

        # チャンネルの変換
        image = image[:, :, ::-1].transpose(2, 0, 1) / 255.0

        # 標準化
        mean = np.array([0.485, 0.456, 0.406], dtype=image.dtype).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=image.dtype).reshape(3, 1, 1)
        image = (image - mean) / std

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

    # 樹種単位で分割する（同一樹種が train/valid 両方に分散しない）
    species_groups = [sample.species for sample in train_all_samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(gss.split(train_all_samples, groups=species_groups))

    train_samples = [train_all_samples[i] for i in train_idx]
    valid_samples = [train_all_samples[i] for i in valid_idx]

    test_samples = load_image_samples(TEST_CSV, TEST_IMAGE_DIR)

    return train_samples, valid_samples, test_samples


def load_split_samples_1d(
    valid_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[WetnessSample], list[WetnessSample], list[WetnessSample]]:
    """train/valid/test の波形サンプル（画像なし）を返す。"""
    train_all_samples = load_csv(TRAIN_CSV)

    # 樹種単位で分割する
    species_groups = [sample.species for sample in train_all_samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(gss.split(train_all_samples, groups=species_groups))

    train_samples = [train_all_samples[i] for i in train_idx]
    valid_samples = [train_all_samples[i] for i in valid_idx]

    test_samples = load_csv(TEST_CSV)

    return train_samples, valid_samples, test_samples

