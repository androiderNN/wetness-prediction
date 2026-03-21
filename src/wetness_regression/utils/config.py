import os
from pathlib import Path
import datetime
from dataclasses import dataclass
import yaml

from wetness_regression.utils.wrpath import OUTPUT_DIR
from wetness_regression.model.regression_model import get_model_input_size


def _parse_yaml_bool(value: object, field_name: str) -> bool:
    """YAML値を厳密に bool へ変換する。"""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False

    raise ValueError(f"{field_name} must be bool")


class output_paths:
    """学習結果の出力に使用するパスの設定"""

    output_dir: Path
    """保存先ディレクトリ"""
    model_path: Path
    """modelのパス"""
    log_path: Path
    """各エポックの損失の保存パス"""
    log_img_path: Path
    """logのプロットのパス"""
    submission_path: Path
    """推論結果のパス"""

    def __init__(self, output_dir: Path):
        os.mkdir(output_dir)
        self.output_dir = output_dir
        self. model_path = output_dir / "model.pth"
        self.log_path = output_dir / "loss.csv"
        self.log_img_path = output_dir / "loss.png"
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
    batch_size: int = 16
    """学習時のバッチサイズ"""
    device: str = "cpu"
    """使用するデバイス"""
    scheduler: str = "linear_decay"
    """学習率スケジューラ名。'none' / 'cosine' / 'warmup_cosine' / 'linear_decay' / 'reduce_on_plateau' / 'step'"""
    freeze_backbone: bool = True
    """True の場合、出力層以外を凍結する"""
    output_dir: Path | None = None
    """出力先の親ディレクトリ"""
    image_size: int = -1
    """入力画像サイズ（正方形、一辺）"""
    paths: output_paths = None
    """出力に関連するパス"""
    use_log_scale: bool = False
    """True の場合、目的変数に log1p/expm1 変換を適用する"""

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = OUTPUT_DIR

        self.image_size = get_model_input_size(self.model_name)

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
    config_dict["batch_size"] = int(config_dict["batch_size"])
    config_dict["freeze_backbone"] = _parse_yaml_bool(config_dict["freeze_backbone"], "freeze_backbone")
    config_dict["use_log_scale"] = _parse_yaml_bool(config_dict.get("use_log_scale", False), "use_log_scale")

    if "output_dir" in config_dict and config_dict["output_dir"] is not None:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return TrainingConfig(**config_dict)
