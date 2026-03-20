from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt

from wetness_regression.utils.wrpath import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
from wetness_regression.dataset.load_dataset import WetnessSample


def plot_sample(sample: WetnessSample):
    """1サンプル分の波形プロットを作成して Figure を返す。"""
    fig = plt.figure(figsize=(6, 6))

    plt.plot(sample.waveform)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis("off")
    plt.ylim(-0.6, 2.1)

    return fig


def make_image(samples: list[WetnessSample]) -> Path:
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
        fig = plot_sample(sample)

        plt.savefig(out_dir / f"{sample.id}.jpg")
        plt.close(fig)
    
    return out_dir

