from typing import Literal

import numpy as np
import numpy.typing as npt


def _resample_to_size(waveform: list[float], size: int) -> npt.NDArray[np.float64]:
	x = np.asarray(waveform, dtype=np.float64)
	x_src = np.linspace(0.0, 1.0, x.size)
	x_dst = np.linspace(0.0, 1.0, size)
	return np.interp(x_dst, x_src, x)


def _scale_01(x: npt.NDArray[np.float64], global_min: float, global_max: float) -> npt.NDArray[np.float64]:
	x = np.clip(x, global_min, global_max)
	return (x - global_min) / (global_max - global_min)


def gramian_angular_field(
	waveform: list[float],
	size: int = 224,
	global_min: float = -0.6,
	global_max: float = 2.1,
	kind: Literal["summation", "difference"] = "summation",
) -> npt.NDArray[np.float32]:
	"""近赤外スペクトルを GAF (GASF/GADF) の 2D 配列へ変換する。"""
	x = _resample_to_size(waveform, size)
	x = _scale_01(x, global_min, global_max)
	x = np.clip(2.0 * x - 1.0, -1.0, 1.0)
	phi = np.arccos(x)

	if kind == "summation":
		m = np.cos(phi[:, None] + phi[None, :])
	else:
		m = np.cos(phi[:, None] - phi[None, :])

	return m.astype(np.float32)


def recurrence_plot(
	waveform: list[float],
	size: int = 224,
	global_min: float = -0.6,
	global_max: float = 2.1,
) -> npt.NDArray[np.float32]:
	"""近赤外スペクトルを Recurrence Plot の 2D 配列へ変換する。"""
	x = _resample_to_size(waveform, size)
	x = _scale_01(x, global_min, global_max)
	m = 1.0 - np.abs(x[:, None] - x[None, :])
	return m.astype(np.float32)


def spectrum_correlation_map(
	waveform: list[float],
	size: int = 224,
	global_min: float = -0.6,
	global_max: float = 2.1,
) -> npt.NDArray[np.float32]:
	"""近赤外スペクトルを外積ベースの相関マップ（2D）へ変換する。"""
	x = _resample_to_size(waveform, size)
	x = _scale_01(x, global_min, global_max)
	x = x - x.mean()
	denom = np.linalg.norm(x) + 1e-8
	x = x / denom
	m = np.outer(x, x)
	return m.astype(np.float32)


def first_derivative_map(
	waveform: list[float],
	size: int = 224,
	global_min: float = -0.6,
	global_max: float = 2.1,
) -> npt.NDArray[np.float32]:
	"""一次微分スペクトルを 2D マップ化して返す。"""
	x = _resample_to_size(waveform, size)
	x = _scale_01(x, global_min, global_max)
	d = np.diff(x, prepend=x[0])
	d = d - d.mean()
	scale = np.max(np.abs(d)) + 1e-8
	d = d / scale
	m = np.outer(d, d)
	return m.astype(np.float32)

