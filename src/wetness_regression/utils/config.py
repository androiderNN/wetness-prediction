from pathlib import Path
import datetime
from dataclasses import dataclass

from wetness_regression.utils.wrpath import OUTPUT_DIR


class output_paths:
    """学習結果の出力に使用するパスの設定"""

    output_dir: Path
    """保存先ディレクトリ"""
    model_path: Path
    """modelのパス"""
    log_path: Path
    """各エポックの損失の保存パス"""
    submission_path: Path
    """推論結果のパス"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self. model_path = output_dir / "model.pth"
        self.log_path = output_dir / "loss.csv"
        self.submission_path = output_dir / "submission.csv"


@dataclass
class TrainingConfig:
    """学習時の設定"""

    model_name: str
    "モデル名"
    num_epochs: int
    """エポック数"""
    lr: float
    """学習率"""
    device: str
    """使用するデバイス"""
    output_dir: Path = OUTPUT_DIR
    """出力先の親ディレクトリ"""
    paths: output_paths = None
    """出力に関連するパス"""

    def __post_init__(self):
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m%d_%H%M%S")
        dirname = f"{timestamp}_{self.model_name}"

        self.paths = output_paths(self.output_dir / dirname)


