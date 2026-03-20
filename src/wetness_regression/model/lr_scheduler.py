from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ConstantLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)


def build_scheduler(name: str, optimizer: Optimizer, num_epochs: int) -> LRScheduler:
    """
    スケジューラ名と総エポック数からスケジューラを構築する。

    Args:
        name: スケジューラ名。以下のいずれかを指定する。
            - "none"          : 学習率を変化させない
            - "cosine"        : CosineAnnealingLR（全体をコサイン減衰）
            - "warmup_cosine" : 先頭 10% をウォームアップ → コサイン減衰（Transformer 向け定番）
            - "linear_decay"  : 進捗 70〜90% の間で線形に 0.1 倍へ漸減
            - "step"          : 全体を 3 等分した間隔で 0.1 倍ずつステップ減衰
        optimizer: 対象の Optimizer
        num_epochs: 総エポック数

    Returns:
        構築されたスケジューラ
    """
    name = name.lower()

    if name == "none":
        return ConstantLR(optimizer, factor=1.0, total_iters=num_epochs)

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    if name == "warmup_cosine":
        warmup_epochs = max(int(0.1 * num_epochs), 1)
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-7)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    if name == "linear_decay":
        milestone1 = int(0.7 * num_epochs)
        milestone2 = int(0.9 * num_epochs)
        decay_epochs = max(milestone2 - milestone1, 1)
        return SequentialLR(
            optimizer,
            schedulers=[
                ConstantLR(optimizer, factor=1.0, total_iters=milestone1),
                LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=decay_epochs),
                ConstantLR(optimizer, factor=0.1, total_iters=num_epochs - milestone2),
            ],
            milestones=[milestone1, milestone2],
        )

    if name == "step":
        step_size = max(num_epochs // 3, 1)
        return StepLR(optimizer, step_size=step_size, gamma=0.1)

    raise ValueError(
        f"scheduler {name} is not defined."
    )
