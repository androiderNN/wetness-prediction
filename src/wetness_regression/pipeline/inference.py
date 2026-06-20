import pandas as pd
import torch
import torch.nn as nn

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.dataset.load_image import WetnessImageSample
from wetness_regression.dataset.load_dataset import WetnessSample
from wetness_regression.pipeline.train import build_image_batch, build_waveform_batch, iter_batches


def inference(
    model: nn.Module,
    cfg: TrainingConfig,
    samples: list[WetnessImageSample] | list[WetnessSample],
    export: bool = False
) -> pd.DataFrame:
    """
    モデルで推論を実行する

    Args:
        model: 学習済みモデル
        cfg: 設定
        samples: テスト用サンプル一覧
        export: submission.csv を作成するかどうか

    Returns:
        result_df: id, pred のDataFrame
    """
    model.to(cfg.device)
    model.eval()

    result = list()

    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size=32, shuffle=False):
            if cfg.model_type.startswith("1d"):
                x = build_waveform_batch(batch_samples)
            else:
                x = build_image_batch(batch_samples, image_size=cfg.image_size)

            x = x.to(cfg.device)
            pred = model(x)

            # マルチタスクモデルは (回帰出力, 分類出力) のタプルを返す
            if isinstance(pred, tuple):
                pred = pred[0]

            for sample, pred_value in zip(batch_samples, pred):
                value = float(pred_value.item())
                if cfg.use_log_scale:
                    value = float(torch.expm1(pred_value).item())
                value = max(value, 0.0)
                result.append([sample.id, value])

    result_df = pd.DataFrame(result, columns=["id", "pred"])

    if export:
        result_df.to_csv(cfg.paths.submission_path, index=False, header=False)

    return result_df
