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
    weight_decay: float = 0.0
    """AdamW の weight decay（L2正則化）の係数"""
    dropout_rate: float = 0.0
    """回帰ヘッドの Dropout 率（0.0 で無効）"""
    use_multi_task: bool = False
    """True の場合、樹種分類を補助タスクとするマルチタスク学習を行う"""
    species_loss_weight: float = 0.5
    """マルチタスク学習時の樹種分類 loss の重み"""
    bottleneck_dim: int = 0
    """回帰ヘッドのボトルネック層の次元数（0 で無効）。768→bottleneck_dim→1 のように中間層を挟む"""
    use_swa: bool = False
    """True の場合、Stochastic Weight Averaging を適用する"""
    swa_start_epoch: int = 0
    """SWA を開始するエポック（0 の場合、全体の 75% 経過後に自動開始）"""
    swa_lr: float = 1e-4
    """SWA 適用中の学習率"""
    use_mixup: bool = False
    """True の場合、MixUp データ拡張を適用する"""
    mixup_alpha: float = 0.2
    """MixUp の Beta 分布パラメータ（小さいほど控えめな混合）"""

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
    config_dict["weight_decay"] = float(config_dict.get("weight_decay", 0.0))
    config_dict["dropout_rate"] = float(config_dict.get("dropout_rate", 0.0))
    config_dict["use_multi_task"] = _parse_yaml_bool(config_dict.get("use_multi_task", False), "use_multi_task")
    config_dict["species_loss_weight"] = float(config_dict.get("species_loss_weight", 0.5))
    config_dict["bottleneck_dim"] = int(config_dict.get("bottleneck_dim", 0))
    config_dict["use_swa"] = _parse_yaml_bool(config_dict.get("use_swa", False), "use_swa")
    config_dict["swa_start_epoch"] = int(config_dict.get("swa_start_epoch", 0))
    config_dict["swa_lr"] = float(config_dict.get("swa_lr", 1e-4))
    config_dict["use_mixup"] = _parse_yaml_bool(config_dict.get("use_mixup", False), "use_mixup")
    config_dict["mixup_alpha"] = float(config_dict.get("mixup_alpha", 0.2))

    if "output_dir" in config_dict and config_dict["output_dir"] is not None:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return TrainingConfig(**config_dict)
