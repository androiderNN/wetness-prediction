from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from wetness_regression.utils.wrpath import TRAIN_CSV, TEST_CSV, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
from wetness_regression.dataset.load_dataset import WetnessSample, load_csv
from wetness_regression.dataset.image_transform import (
    first_derivative_map,
    gramian_angular_field,
    recurrence_plot,
    spectrum_correlation_map,
)


METHOD_TO_FN = {
    "gaf": gramian_angular_field,
    "recurrence": recurrence_plot,
    "correlation": spectrum_correlation_map,
    "derivative": first_derivative_map,
}


def plot_sample(sample: WetnessSample, figsize: tuple[int] = (6, 6)):
    """1サンプル分の波形プロットを作成して Figure を返す。"""
    fig = plt.figure(figsize=figsize)

    plt.plot(sample.waveform)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis("off")
    plt.ylim(-0.6, 2.1)

    return fig


def make_image(samples: list[WetnessSample], figsize: tuple[float] = (2.24, 2.24), dpi: int = 100) -> Path:
    """
    データを画像として保存する

    Args:
        samples: 計測データの配列
        export_dir: 保存先の親ディレクトリ

    Returns:
        out_dir: 保存先のディレクトリ
    """
    out_dir = TEST_IMAGE_DIR if samples[0].target is None else TRAIN_IMAGE_DIR

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)

    for sample in samples:
        fig = plot_sample(sample, figsize)

        plt.savefig(out_dir / f"{sample.id}.png", dpi=dpi)
        plt.close(fig)

    return out_dir


def _normalize_channel(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """2D配列を [0, 1] に正規化する。"""
    x = x.astype(np.float32)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if vmax - vmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - vmin) / (vmax - vmin)


def build_three_channel_image(
    waveform: list[float],
    methods: tuple[str, str, str],
    size: int = 224,
) -> npt.NDArray[np.uint8]:
    """指定した3手法で3ch画像を構築する。"""
    channels = []
    for method in methods:
        if method not in METHOD_TO_FN:
            raise ValueError(f"Inknown method: {method}")
        channel = METHOD_TO_FN[method](waveform, size=size)
        channels.append(_normalize_channel(channel))

    image = np.stack(channels, axis=2)  # (H, W, 3)
    return (image * 255.0).astype(np.uint8)


def make_image_three_channel(
    samples: list[WetnessSample],
    methods: tuple[str, str, str] = ("gaf", "recurrence", "derivative"),
    size: int = 224,
) -> Path:
    """サンプル一覧を 3ch 画像に変換して保存する。"""
    out_dir = TEST_IMAGE_DIR if samples[0].target is None else TRAIN_IMAGE_DIR

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for sample in samples:
        img = build_three_channel_image(sample.waveform, methods=methods, size=size)
        plt.imsave(out_dir / f"{sample.id}.png", img)

    return out_dir


if __name__ == '__main__':
    # 波形画像の作成
    samples = load_csv(TRAIN_CSV)
    _ = make_image_three_channel(samples)

    samples = load_csv(TEST_CSV)
    _ = make_image_three_channel(samples)
