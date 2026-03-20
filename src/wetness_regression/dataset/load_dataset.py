from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader


@dataclass
class WetnessSample:
    """データ一件分を保持する"""

    id: int
    """sample number"""
    species: int
    """species number"""
    species_japanese: str
    """樹種"""
    waveform: list[int]
    """測定データのリスト"""
    target: float | None = None
    """含水率（学習データのみ）"""


def load_csv(csv_path: Path) -> list[WetnessSample]:
    df = pd.read_csv(csv_path, encoding="shift-jis")

    has_target = "含水率" in df.columns
    feature_rows = df.iloc[:, -1555:].to_numpy(copy=False).tolist()

    meta_columns = ["sample number", "species number", "樹種"]
    if has_target:
        meta_columns.append("含水率")

    meta_rows = df.loc[:, meta_columns].to_dict("records")

    samples = []

    for meta_row, feature_row in zip(meta_rows, feature_rows):
        sample = WetnessSample(
            id=meta_row["sample number"],
            species=meta_row["species number"],
            species_japanese=meta_row["樹種"],
            waveform=feature_row,
            target=meta_row["含水率"] if has_target else None,
        )
        samples.append(sample)

    return samples
