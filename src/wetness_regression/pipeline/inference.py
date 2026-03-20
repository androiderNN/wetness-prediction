import pandas as pd
import torch
from torch.utils.data import DataLoader

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel
from wetness_regression.dataset.load_image import WetnessImageDataset


def inference(
    model: RegressionModel,
    cfg: TrainingConfig,
    dataloader: DataLoader,
    export: bool = False
) -> pd.DataFrame:
    """
    モデルで推論を実行する

    Args:
        model: 学習済みモデル
        cfg: 設定
        dataloader: テストデータの DataLoader
        export: submission.csv を作成するかどうか
    
    Returns:
        result_df: id, pred のDataFrame
    """
    model.to(cfg.device)
    model.eval()

    result = list()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(cfg.device)
            pred = model(images)

            # バッチ内の各サンプルの予測値を結果に追加
            for i in range(images.shape[0]):
                result.append([batch_idx * dataloader.batch_size + i, float(pred[i])])
    
    result_df = pd.DataFrame(result, columns=["id", "pred"])

    if export:
        result_df.to_csv(cfg.paths.submission_path, index=False, header=False)
    
    return result_df
