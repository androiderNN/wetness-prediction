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


def build_three_channel_image(
    waveform: list[float],
    methods: tuple[str, str, str],
    size: int = 224,
    channel_minmax: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> npt.NDArray[np.uint8]:
    """指定した3手法で3ch画像を構築する。"""
    channels = []
    for idx, method in enumerate(methods):
        if method not in METHOD_TO_FN:
            raise ValueError(f"Unknown method: {method}")
        channel = METHOD_TO_FN[method](waveform, size=size)

        if channel_minmax is None:
            vmin = float(np.min(channel))
            vmax = float(np.max(channel))
        else:
            vmin, vmax = channel_minmax[idx]

        if vmax - vmin < 1e-8:
            channel = np.zeros_like(channel, dtype=np.float32)
        else:
            channel = np.clip(channel, vmin, vmax)
            channel = (channel - vmin) / (vmax - vmin)

        channels.append(channel)

    image = np.stack(channels, axis=2)  # (H, W, 3)

    return (image * 255.0).astype(np.uint8)


def compute_global_channel_minmax(
    samples: list[WetnessSample],
    methods: tuple[str, str, str],
    size: int = 224,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """全サンプルを走査して、各チャンネルの最小値・最大値を返す。"""
    mins = [float("inf")] * len(methods)
    maxs = [float("-inf")] * len(methods)

    for sample in samples:
        for idx, method in enumerate(methods):
            if method not in METHOD_TO_FN:
                raise ValueError(f"Unknown method: {method}")

            channel = METHOD_TO_FN[method](sample.waveform, size=size)
            cmin = float(np.min(channel))
            cmax = float(np.max(channel))

            if cmin < mins[idx]:
                mins[idx] = cmin
            if cmax > maxs[idx]:
                maxs[idx] = cmax

    return (
        (mins[0], maxs[0]),
        (mins[1], maxs[1]),
        (mins[2], maxs[2]),
    )


def make_image_three_channel(
    samples: list[WetnessSample],
    methods: tuple[str, str, str] = ("gaf", "recurrence", "derivative"),
    size: int = 224,
    channel_minmax: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> Path:
    """サンプル一覧を 3ch 画像に変換して保存する。"""
    out_dir = TEST_IMAGE_DIR if samples[0].target is None else TRAIN_IMAGE_DIR

    if channel_minmax is None:
        channel_minmax = compute_global_channel_minmax(samples, methods=methods, size=size)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for sample in samples:
        img = build_three_channel_image(
            sample.waveform,
            methods=methods,
            size=size,
            channel_minmax=channel_minmax,
        )
        plt.imsave(out_dir / f"{sample.id}.png", img)

    return out_dir


if __name__ == '__main__':
    methods = ("gaf", "recurrence", "derivative")
    train_samples = load_csv(TRAIN_CSV)
    test_samples = load_csv(TEST_CSV)

    # train/test の両方を使ってグローバル min/max を揃える
    channel_minmax = compute_global_channel_minmax(train_samples + test_samples, methods=methods)

    _ = make_image_three_channel(train_samples, methods=methods, channel_minmax=channel_minmax)
    _ = make_image_three_channel(test_samples, methods=methods, channel_minmax=channel_minmax)
