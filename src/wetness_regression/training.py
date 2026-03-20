import sys
import argparse
import shutil
from pathlib import Path

from wetness_regression.utils.config import load_trainingconfig
from wetness_regression.dataset.load_image import load_split_samples
from wetness_regression.pipeline.train import train
from wetness_regression.pipeline.inference import inference


def run_training(config_path: Path):
    """
    yaml設定ファイルを読み込んで学習と推論を実行する

    Args:
        config_path: YAML設定ファイルのパス
    """
    cfg = load_trainingconfig(config_path)
    print(f"Loading config from: {config_path}")

    # yaml設定ファイルを出力ディレクトリにコピー
    config_copy_path = cfg.paths.output_dir / config_path.name
    shutil.copy(config_path, config_copy_path)

    # 学習用・検証用・推論用サンプルを構築
    train_samples, valid_samples, test_samples = load_split_samples()

    # 学習を実行
    model = train(cfg, train_samples, valid_samples)

    # 推論を実行
    result_df = inference(model, cfg, test_samples, export=True)
    print(result_df.head())


def main():
    parser = argparse.ArgumentParser(
        description="含水率回帰モデルの学習スクリプト"
    )
    parser.add_argument(
        "config",
        type=str,
        help="学習設定ファイル（yaml）のパス",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"エラー: 設定ファイルが見つかりません: {config_path}", file=sys.stderr)
        sys.exit(1)

    run_training(config_path)


if __name__ == "__main__":
    main()
