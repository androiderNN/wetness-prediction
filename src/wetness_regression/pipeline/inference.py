import pandas as pd
import torch

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel
from wetness_regression.dataset.load_image import WetnessImageSample
from wetness_regression.pipeline.train import build_image_batch, iter_batches


def inference(
    model: RegressionModel,
    cfg: TrainingConfig,
    samples: list[WetnessImageSample],
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
    image_size = cfg.image_size

    result = list()

    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size=32, shuffle=False):
            images = build_image_batch(batch_samples, image_size=image_size)
            images = images.to(cfg.device)
            pred = model(images)

            for sample, pred_value in zip(batch_samples, pred):
                result.append([sample.id, float(pred_value.item())])
    
    result_df = pd.DataFrame(result, columns=["id", "pred"])

    if export:
        result_df.to_csv(cfg.paths.submission_path, index=False, header=False)
    
    return result_df
