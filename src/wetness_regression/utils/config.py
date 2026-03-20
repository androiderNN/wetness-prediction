import os
from pathlib import Path
import datetime
from dataclasses import dataclass
import yaml

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
        os.mkdir(output_dir)
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
    output_dir: Path | None
    """出力先の親ディレクトリ"""
    paths: output_paths = None
    """出力に関連するパス"""

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = OUTPUT_DIR

        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m%d_%H%M%S")
        dirname = f"{timestamp}_{self.model_name}"

        self.paths = output_paths(self.output_dir / dirname)


def load_trainingconfig(yaml_path: Path | str) -> TrainingConfig:
    """
    yamlファイルから TrainingConfig を読み込む

    Args:
        yaml_path: yamlファイルのパス

    Returns:
        TrainingConfig: 読み込まれた設定
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not exists.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError("yamlファイルが空です")

    # 型変換
    config_dict["num_epochs"] = int(config_dict["num_epochs"])
    config_dict["lr"] = float(config_dict["lr"])

    if "output_dir" in config_dict and config_dict["output_dir"] is not None:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return TrainingConfig(**config_dict)
