import random
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wetness_regression.model.lr_scheduler import build_scheduler

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel, RMSELoss
from wetness_regression.dataset.load_image import WetnessImageSample


def build_image_batch(samples: list[WetnessImageSample], image_size: int) -> torch.Tensor:
    """サンプル一覧から画像テンソルを構築する。"""
    images = [torch.from_numpy(sample.image).float() for sample in samples]
    batch = torch.stack(images)

    # モデル入力サイズに合わせたリサイズ
    current_size = images[0].shape[1]

    if current_size != image_size:
        print(f"reshaping images from {current_size} to {image_size}")
        batch = F.interpolate(batch, size=(image_size, image_size), mode="bilinear", align_corners=False)

    return batch


def build_target_batch(samples: list[WetnessImageSample], use_log_scale: bool) -> torch.Tensor:
    """サンプル一覧から教師信号テンソルを構築する。"""
    targets = [float(sample.target) for sample in samples]
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    if use_log_scale:
        y = torch.log1p(y)
    return y


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
    cfg: TrainingConfig,
    model: RegressionModel,
    samples: list[WetnessImageSample],
    criterion: nn.Module,
) -> tuple[float, float | None]:
    """
    サンプル一覧に対する平均損失を計算する。
    use_log_scale=Trueの場合、元スケールでのRMSEも計算する。

    Returns:
        (loss_log_space, loss_original_scale)
        use_log_scale=Falseの場合、loss_original_scaleはNone
    """
    model.eval()
    running_loss = 0.0
    squared_errors_orig = [] if cfg.use_log_scale else None

    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size=cfg.batch_size, shuffle=False):
            x = build_image_batch(batch_samples, image_size=cfg.image_size)
            y = build_target_batch(batch_samples, use_log_scale=cfg.use_log_scale)
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss.item() * len(batch_samples)

            # use_log_scale=Trueの場合、元スケール損失も計算
            if cfg.use_log_scale and squared_errors_orig is not None:
                y_orig = torch.tensor([float(s.target) for s in batch_samples], dtype=torch.float32).unsqueeze(1)
                y_orig = y_orig.to(cfg.device)
                pred_orig = torch.expm1(pred)
                errors = pred_orig - y_orig
                squared_errors_orig.extend((errors ** 2).cpu().numpy().flatten())

    loss_log = running_loss / len(samples)
    loss_orig = float(np.sqrt(np.mean(squared_errors_orig))) if squared_errors_orig is not None else None

    return loss_log, loss_orig


def train(
    cfg: TrainingConfig,
    train_samples: list[WetnessImageSample],
    valid_samples: list[WetnessImageSample],
):
    """
    学習を実行する

    Args:
        cfg: 学習設定
        train_samples: 学習用サンプル
        valid_samples: 検証用サンプル
        batch_size: バッチサイズ
    """
    model = RegressionModel(
        pretrained_model_name=cfg.model_name,
        freeze_backbone=cfg.freeze_backbone,
    )
    model.to(cfg.device)

    # 必要なパラメータのみ学習する
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = RMSELoss()
    scheduler = build_scheduler(cfg.scheduler, optimizer, cfg.num_epochs)

    print(f"\nStarting training.")
    print(f"config: {asdict(cfg)}\n")

    loss_log = list()
    best_epoch = -1
    best_loss = float("inf")
    best_model = None

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_samples in iter_batches(train_samples, batch_size=cfg.batch_size, shuffle=True):
            x = build_image_batch(batch_samples, image_size=cfg.image_size)
            y = build_target_batch(batch_samples, use_log_scale=cfg.use_log_scale)
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_samples)

        scheduler.step()

        # lossの計算
        epoch_train_loss = running_loss / len(train_samples)

        epoch_valid_loss, epoch_valid_loss_orig = evaluate(
            cfg,
            model,
            valid_samples,
            criterion=criterion,
        )

        loss_log.append([epoch, epoch_train_loss, epoch_valid_loss])
        print(f"Epoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}")

        if cfg.use_log_scale and epoch_valid_loss_orig is not None:
            print(f"Valid Loss (original scale): {epoch_valid_loss_orig:.4f})")

        # ログの更新
        log_df = pd.DataFrame(loss_log)
        log_df.columns = ["epoch", "train", "valid"]
        log_df.to_csv(cfg.paths.log_path, index=False)

        # ロスが下がった場合はモデルを更新
        if epoch_valid_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_valid_loss
            best_model = model.state_dict()

            # モデルの保存
            torch.save(best_model, cfg.paths.model_path)

    print(f"\ntraining finished.\nBest Epoch: {best_epoch} loss: {best_loss}")

    # プロット
    log_df = log_df.iloc[:, 1:]
    plt.plot(log_df, label=log_df.columns.tolist())
    plt.legend()
    plt.savefig(cfg.paths.log_img_path)

    model.load_state_dict(best_model)
    return model
