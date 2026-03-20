from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from wetness_regression.utils.config import TrainingConfig
from wetness_regression.model.regression_model import RegressionModel


def train(cfg: TrainingConfig, train_dataloader: DataLoader, valid_dataloader: DataLoader):
    """
    学習を実行する

    Args:
        cfg: 学習設定
        dataloader: DataLoader
    """
    model = RegressionModel(pretrained_model_name=cfg.model_name)
    model.to(cfg.device)
    model.train() # Set model to training mode

    # 必要なパラメータのみ学習する
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    # TODO: 損失関数の選択
    criterion = nn.MSELoss()

    print(f"Starting fine-tuning.")
    print(f"config: {asdict(cfg)}")

    loss_log = list()
    best_epoch = -1
    best_loss = float("inf")
    best_model = None

    for epoch in range(cfg.num_epochs):
        # 学習ステップ
        model.train()
        running_loss = 0.0

        for x, y in train_dataloader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            # 勾配の初期化
            optimizer.zero_grad()

            pred = model(x)
            # targets = targets.float()

            loss = criterion(pred, y)

            # パラメータ更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)

        # 検証ステップ
        model.eval()
        valid_running_loss = 0.0

        with torch.no_grad():
            for x, y in valid_dataloader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                pred = model(x)
                loss = criterion(pred, y)
                valid_running_loss += loss.item() * x.size(0)

        epoch_valid_loss = valid_running_loss / len(valid_dataloader.dataset)

        print(f"Epoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}")

        loss_log.append([epoch, epoch_train_loss, epoch_valid_loss])

        # ロスが下がった場合はモデルを更新
        if epoch_valid_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_valid_loss
            best_model = model.state_dict()

    print("training complete.")
    print(f"Best Epoch: {best_epoch} loss: {best_loss}")

    # 学習結果の保存
    torch.save(best_model, cfg.paths.model_path)

    log = pd.DataFrame(loss_log)
    log.to_csv(cfg.paths.log_path)

    model = model.load_state_dict(best_model)
    return model
