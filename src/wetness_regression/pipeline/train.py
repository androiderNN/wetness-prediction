import random
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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
) -> tuple[float, float]:
    """
    サンプル一覧に対するRMSEを正しく計算する。
    全サンプルの二乗誤差を蓄積してから sqrt を取るため、
    バッチ分割に依存しない正確な値を返す。

    Returns:
        (loss_log_space, loss_original_scale)
        use_log_scale=Falseの場合、両者は同じ値
    """
    model.eval()
    squared_errors_log = []
    squared_errors_orig = []

    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size=cfg.batch_size, shuffle=False):
            x = build_image_batch(batch_samples, image_size=cfg.image_size)
            y = build_target_batch(batch_samples, use_log_scale=cfg.use_log_scale)
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)

            # log空間の二乗誤差を蓄積
            errors_log = pred - y
            squared_errors_log.extend((errors_log ** 2).cpu().numpy().flatten())

            # 元スケールの二乗誤差を蓄積
            if cfg.use_log_scale:
                pred_orig = torch.expm1(pred)
                y_orig = torch.tensor(
                    [float(s.target) for s in batch_samples], dtype=torch.float32
                ).unsqueeze(1).to(cfg.device)
                errors_orig = pred_orig - y_orig
            else:
                errors_orig = errors_log
            squared_errors_orig.extend((errors_orig ** 2).cpu().numpy().flatten())

    loss_log = float(np.sqrt(np.mean(squared_errors_log)))
    loss_orig = float(np.sqrt(np.mean(squared_errors_orig)))

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
        )

        loss_log.append([epoch, epoch_train_loss, epoch_valid_loss_orig])
        print(f"Epoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss_orig:.4f}")
        if cfg.use_log_scale:
            print(f"Valid Loss (log scale): {epoch_valid_loss:.4f}")

        # ログの更新
        log_df = pd.DataFrame(loss_log)
        log_df.columns = ["epoch", "train", "valid"]
        log_df.to_csv(cfg.paths.log_path, index=False)

        # ロスが下がった場合はモデルを更新（元スケールRMSEで判定）
        if epoch_valid_loss_orig < best_loss:
            best_epoch = epoch
            best_loss = epoch_valid_loss_orig
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
