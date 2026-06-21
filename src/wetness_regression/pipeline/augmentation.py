import torch

from wetness_regression.utils.config import TrainingConfig


def apply_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """ガウスノイズを付加する。1D (B,T) / 2D (B,C,H,W) 両対応。"""
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def apply_random_wavelength_shift(x: torch.Tensor, shift_max: int) -> torch.Tensor:
    """波長軸方向にランダムシフト。1D は dim=-1、2D は dim=-1（水平方向）に torch.roll。"""
    if shift_max <= 0:
        return x
    batch_size = x.size(0)
    shifts = torch.randint(-shift_max, shift_max + 1, (batch_size,), device=x.device)
    # バッチ内の各サンプルを個別のシフト量でロール
    shifted = []
    for i in range(batch_size):
        shifted.append(torch.roll(x[i], shifts[i].item(), dims=-1))
    return torch.stack(shifted)


def apply_random_scaling(
    x: torch.Tensor, scale_range: tuple[float, float], is_1d: bool
) -> torch.Tensor:
    """ランダムスケーリング。1D は全体一括、2D はチャンネル独立。"""
    lo, hi = scale_range
    if lo >= hi:
        return x
    if is_1d:
        # (B, T): サンプルごとに1つのスケール係数
        scales = torch.empty(x.size(0), 1, device=x.device).uniform_(lo, hi)
    else:
        # (B, C, H, W): サンプル×チャンネルごとに独立スケール
        scales = torch.empty(x.size(0), x.size(1), 1, 1, device=x.device).uniform_(lo, hi)
    return x * scales


def apply_baseline_shift(
    x: torch.Tensor, shift_max: float, num_segments: int, is_1d: bool
) -> torch.Tensor:
    """ベースラインシフト。1D はセグメント分割、2D はチャンネルごと定数オフセット。"""
    if shift_max <= 0:
        return x
    if is_1d:
        # x: (B, T) を num_segments に分割し、セグメントごとに異なるオフセット
        B, T = x.shape
        num_segments = min(num_segments, T)  # T より大きい場合はクランプ
        seg_len = max(1, T // num_segments)
        offsets = torch.empty(B, num_segments, device=x.device).uniform_(-shift_max, shift_max)
        x = x.clone()  # 入力テンソルを破壊しないようコピー
        for s in range(num_segments):
            start = s * seg_len
            end = start + seg_len if s < num_segments - 1 else T
            x[:, start:end] += offsets[:, s:s+1]
    else:
        # 2D: チャンネルごとに定数オフセット (B, C, 1, 1)
        offsets = torch.empty(x.size(0), x.size(1), 1, 1, device=x.device).uniform_(-shift_max, shift_max)
        x = x + offsets
    return x


def apply_augmentations(x: torch.Tensor, cfg: TrainingConfig, is_1d: bool) -> torch.Tensor:
    """全オーグメンテーションを順に適用。cfg.use_augmentation=False の場合はスキップ。"""
    if not cfg.use_augmentation:
        return x

    scale_range = tuple(cfg.aug_scale_range) if cfg.aug_scale_range else (0.95, 1.05)

    x = apply_random_wavelength_shift(x, cfg.aug_shift_max)
    x = apply_random_scaling(x, scale_range, is_1d)
    x = apply_baseline_shift(x, cfg.aug_baseline_shift_max, cfg.aug_baseline_num_segments, is_1d)
    x = apply_gaussian_noise(x, cfg.aug_noise_std)
    return x
