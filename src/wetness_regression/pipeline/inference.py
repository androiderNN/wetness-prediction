import pandas as pd
import torch

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel
from wetness_regression.dataset.load_image import WetnessImageDataset


def inference(
    model: RegressionModel,
    cfg: TrainingConfig,
    samples: list[WetnessImageDataset],
    export: bool = False
) -> pd.DataFrame:
    """
    モデルで推論＆出力する

    Args:
        model: モデル
        cfg: 設定
        dataloader: DataLoader
        export: submission.csvを作成するかどうか
    
    Returns:
        result_df: id, predのdataframe
    """
    model.to(cfg.device)
    model.eval()

    result = list()

    for sample in samples:
        id = sample.id
        x = torch.tensor(sample.image, dtype=torch.float32)

        pred = model(x)

        result.append([id, float(pred)])
    
    result_df = pd.DataFrame(result)

    if export:
        result_df.to_csv(cfg.paths.submission_path)
    
    return result_df
