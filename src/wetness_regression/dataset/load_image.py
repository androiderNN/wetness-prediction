import os
import math
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
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

        return cls(**wnsample, image=cv2.imread(img_path))


class WetnessImageDataset(Dataset):
    """学習に使う画像と目的値を返すDataset。"""

    def __init__(self, samples: list[WetnessImageSample], require_target: bool = True):
        self.samples = samples
        self.require_target = require_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        image = torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0
        target_value = math.nan if sample.target is None else float(sample.target)
        target = torch.tensor([target_value], dtype=torch.float32)

        return image, target


def build_dataloader(
    batch_size: int = 16,
    test_size: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("valid_ratio は 0.0 より大きく 1.0 より小さい必要があります。")

    train_samples = load_csv(TRAIN_CSV)
    train_img_samples = [WetnessImageSample.from_wetnesssample(s, TRAIN_IMAGE_DIR) for s in train_samples]
    
    tra_img_samples, val_img_samples = train_test_split(train_img_samples, test_size=test_size, random_state=seed, shuffle=True)

    train_dataset = WetnessImageDataset(tra_img_samples, require_target=True)
    valid_dataset = WetnessImageDataset(val_img_samples, require_target=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, valid_dataloader

    test_samples = load_csv(TEST_CSV)
    test_img_samples = [WetnessImageSample.from_wetnesssample(s, TEST_IMAGE_DIR) for s in test_samples]
    test_dataset = WetnessImageDataset(test_img_samples, require_target=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, valid_dataloader, test_dataloader

