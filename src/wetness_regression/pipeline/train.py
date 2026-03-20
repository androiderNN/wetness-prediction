import random
from dataclasses import asdict
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel, RMSELoss
from wetness_regression.dataset.load_image import WetnessImageSample


def build_image_batch(samples: list[WetnessImageSample]) -> torch.Tensor:
    """サンプル一覧から画像テンソルを構築する。"""
    images = [torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0 for sample in samples]
    return torch.stack(images)


def build_target_batch(samples: list[WetnessImageSample]) -> torch.Tensor:
    """サンプル一覧から教師信号テンソルを構築する。"""
    targets = [float(sample.target) for sample in samples]
    return torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def iter_batches(
    samples: list[WetnessImageSample],
    batch_size: int,
    shuffle: bool,
) -> list[list[WetnessImageSample]]:
    """サンプル一覧を固定サイズのバッチへ分割する。"""
    ordered_samples = list(samples)
    if shuffle:
        random.shuffle(ordered_samples)

    return [ordered_samples[i:i + batch_size] for i in range(0, len(ordered_samples), batch_size)]


def evaluate(
    model: RegressionModel,
    samples: list[WetnessImageSample],
    batch_size: int,
    criterion: nn.Module,
    device: str,
) -> float:
    """サンプル一覧に対する平均損失を計算する。"""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size=batch_size, shuffle=False):
            x = build_image_batch(batch_samples)
            y = build_target_batch(batch_samples)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss.item() * len(batch_samples)

    return running_loss / len(samples)


def train(
    cfg: TrainingConfig,
    train_samples: list[WetnessImageSample],
    valid_samples: list[WetnessImageSample],
    batch_size: int = 16,
):
    """
    学習を実行する

    Args:
        cfg: 学習設定
        train_samples: 学習用サンプル
        valid_samples: 検証用サンプル
        batch_size: バッチサイズ
    """
    model = RegressionModel(pretrained_model_name=cfg.model_name)
    model.to(cfg.device)

    # 必要なパラメータのみ学習する
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = RMSELoss()

    print(f"\nStarting training.")
    print(f"config: {asdict(cfg)}\n")

    loss_log = list()
    best_epoch = -1
    best_loss = float("inf")
    best_model = None

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_samples in iter_batches(train_samples, batch_size=batch_size, shuffle=True):
            x = build_image_batch(batch_samples)
            y = build_target_batch(batch_samples)
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_samples)

        epoch_train_loss = running_loss / len(train_samples)
        epoch_valid_loss = evaluate(
            model,
            valid_samples,
            batch_size=batch_size,
            criterion=criterion,
            device=cfg.device,
        )

        print(f"Epoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}")

        loss_log.append([epoch, epoch_train_loss, epoch_valid_loss])

        # ロスが下がった場合はモデルを更新
        if epoch_valid_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_valid_loss
            best_model = model.state_dict()

    print(f"\ntraining finished.\nBest Epoch: {best_epoch} loss: {best_loss}")

    # 学習結果の保存
    torch.save(best_model, cfg.paths.model_path)

    log = pd.DataFrame(loss_log)
    log.columns = ["epoch", "train", "valid"]
    log.to_csv(cfg.paths.log_path, index=False)

    # プロット
    log = log.iloc[:, 1:]
    plt.plot(log, label=log.columns.tolist())
    plt.legend()
    plt.savefig(cfg.paths.log_img_path)

    model.load_state_dict(best_model)
    return model
