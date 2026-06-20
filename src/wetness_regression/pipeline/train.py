import random
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from wetness_regression.model.lr_scheduler import build_scheduler

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel, RMSELoss
from wetness_regression.model.multi_task_model import MultiTaskModel
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


def build_species_batch(
    samples: list[WetnessImageSample],
    species_to_idx: dict[int, int],
) -> torch.Tensor:
    """サンプル一覧から樹種ラベルのテンソルを構築する（0-indexed のクラスインデックス）。"""
    indices = [species_to_idx[sample.species] for sample in samples]
    return torch.tensor(indices, dtype=torch.long)


def apply_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    sp: torch.Tensor | None,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """MixUp データ拡張を適用する。y は元スケール（log変換前）のターゲット。
    sp が None でない場合はソフトラベル（確率分布）を返す。"""
    if alpha <= 0:
        return x, y, sp

    batch_size = x.size(0)
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    mixed_sp = None
    if sp is not None:
        num_classes = int(sp.max().item()) + 1
        sp_onehot = torch.zeros(batch_size, num_classes, device=x.device)
        sp_onehot.scatter_(1, sp.unsqueeze(1), 1.0)
        sp_idx = torch.zeros(batch_size, num_classes, device=x.device)
        sp_idx.scatter_(1, sp[index].unsqueeze(1), 1.0)
        mixed_sp = lam * sp_onehot + (1 - lam) * sp_idx

    return mixed_x, mixed_y, mixed_sp


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

            # マルチタスクモデルは (回帰出力, 分類出力) のタプルを返す
            if isinstance(pred, tuple):
                pred = pred[0]

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
    # 樹種→クラスインデックスのマッピングを構築
    species_to_idx: dict[int, int] | None = None
    if cfg.use_multi_task:
        unique_species = sorted({s.species for s in train_samples})
        species_to_idx = {sp: idx for idx, sp in enumerate(unique_species)}
        num_species = len(unique_species)
        print(f"Multi-task mode: {num_species} species classes")

        model = MultiTaskModel(
            pretrained_model_name=cfg.model_name,
            num_species=num_species,
            freeze_backbone=cfg.freeze_backbone,
            dropout_rate=cfg.dropout_rate,
            bottleneck_dim=cfg.bottleneck_dim,
        )
    else:
        model = RegressionModel(
            pretrained_model_name=cfg.model_name,
            freeze_backbone=cfg.freeze_backbone,
            dropout_rate=cfg.dropout_rate,
            bottleneck_dim=cfg.bottleneck_dim,
        )
    model.to(cfg.device)

    # 必要なパラメータのみ学習する
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = RMSELoss()
    cls_criterion = torch.nn.CrossEntropyLoss() if cfg.use_multi_task else None
    scheduler = build_scheduler(cfg.scheduler, optimizer, cfg.num_epochs)

    # SWA のセットアップ
    swa_model: AveragedModel | None = None
    swa_scheduler: SWALR | None = None
    swa_start = 0
    if cfg.use_swa:
        swa_model = AveragedModel(model)
        swa_start = cfg.swa_start_epoch if cfg.swa_start_epoch > 0 else int(0.75 * cfg.num_epochs)
        print(f"SWA enabled: start epoch {swa_start}/{cfg.num_epochs}, swa_lr={cfg.swa_lr}")

    print(f"\nStarting training.")
    print(f"config: {asdict(cfg)}\n")

    loss_log = list()
    best_epoch = -1
    best_loss = float("inf")
    best_model = None

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0

        for batch_samples in iter_batches(train_samples, batch_size=cfg.batch_size, shuffle=True):
            x = build_image_batch(batch_samples, image_size=cfg.image_size)

            # MixUp を使う場合は元スケールでターゲットを構築し、混合後に log 変換
            need_original_for_mixup = cfg.use_mixup and cfg.use_log_scale
            y = build_target_batch(batch_samples, use_log_scale=(cfg.use_log_scale and not need_original_for_mixup))
            x, y = x.to(cfg.device), y.to(cfg.device)

            if cfg.use_multi_task:
                sp = build_species_batch(batch_samples, species_to_idx).to(cfg.device)
            else:
                sp = None

            # MixUp の適用（元スケール）
            if cfg.use_mixup:
                x, y, sp = apply_mixup(x, y, sp, cfg.mixup_alpha)
                if need_original_for_mixup:
                    y = torch.log1p(y.clamp(min=0))

            optimizer.zero_grad()

            if cfg.use_multi_task:
                pred_reg, pred_cls = model(x)
                loss_reg = criterion(pred_reg, y)
                if cfg.use_mixup:
                    # ソフトラベルとの交差エントロピー
                    loss_cls = -(sp * F.log_softmax(pred_cls, dim=1)).sum(dim=1).mean()
                else:
                    loss_cls = cls_criterion(pred_cls, sp)
                loss = loss_reg + cfg.species_loss_weight * loss_cls
                running_cls_loss += loss_cls.item() * len(batch_samples)
            else:
                pred = model(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_samples)

        # SWA モード: 定常LR + 重みの移動平均を更新
        if cfg.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler is None:
                swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
            swa_scheduler.step()
        else:
            scheduler.step()

        # lossの計算
        epoch_train_loss = running_loss / len(train_samples)

        epoch_valid_loss, epoch_valid_loss_orig = evaluate(
            cfg,
            model,
            valid_samples,
        )

        loss_log.append([epoch, epoch_train_loss, epoch_valid_loss_orig])
        train_info = f"Epoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_train_loss:.4f}"
        if cfg.use_multi_task:
            epoch_train_cls_loss = running_cls_loss / len(train_samples)
            train_info += f", Train Cls Loss: {epoch_train_cls_loss:.4f}"
        print(f"{train_info}, Valid Loss: {epoch_valid_loss_orig:.4f}")
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

    # SWA モデルの評価
    if swa_model is not None:
        # BN 統計の更新
        swa_model.train()
        with torch.no_grad():
            for batch_samples in iter_batches(train_samples, batch_size=cfg.batch_size, shuffle=False):
                x = build_image_batch(batch_samples, image_size=cfg.image_size)
                x = x.to(cfg.device)
                swa_model(x)

        _, swa_valid_loss_orig = evaluate(cfg, swa_model, valid_samples)
        swa_better = swa_valid_loss_orig < best_loss
        print(f"SWA model - Valid Loss: {swa_valid_loss_orig:.4f} {'(better)' if swa_better else ''}")

        if swa_better:
            best_model = swa_model.state_dict()
            best_loss = swa_valid_loss_orig
            torch.save(best_model, cfg.paths.model_path)

    # プロット
    log_df = log_df.iloc[:, 1:]
    plt.plot(log_df, label=log_df.columns.tolist())
    plt.legend()
    plt.savefig(cfg.paths.log_img_path)

    # 最終モデルをロードして返す
    final_model = swa_model if (swa_model is not None and swa_better) else model
    if not (swa_model is not None and swa_better):
        final_model.load_state_dict(best_model)
    return final_model
