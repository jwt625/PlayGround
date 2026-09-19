"""Paper target loading and 1-dB crossing analysis (T02/T09).

Relative and absolute 1-dB definitions are both supported because Table 2's
heading is ambiguous (PAPER_REVIEW.md s6 item 5): the main reproduction
convention is relative to the peak (0.88 -> 0.699), with the absolute
definition (0.794) exported separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_tables(path: str | Path = "data/paper_targets/tables.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _crossing(wl: np.ndarray, eta: np.ndarray, level: float, side: str) -> float | None:
    """First monotone crossing of ``eta = level`` on the given side of the peak."""
    ipk = int(np.argmax(eta))
    if side == "left":
        seg_wl, seg_eta = wl[: ipk + 1][::-1], eta[: ipk + 1][::-1]
    else:
        seg_wl, seg_eta = wl[ipk:], eta[ipk:]
    for i in range(len(seg_eta) - 1):
        a, b = seg_eta[i] - level, seg_eta[i + 1] - level
        if a == 0:
            return float(seg_wl[i])
        if a * b < 0:
            t = a / (a - b)
            return float(seg_wl[i] + t * (seg_wl[i + 1] - seg_wl[i]))
    return None


def peak_and_edges(wl_nm, eta, definition: str = "relative", drop_db: float = 1.0) -> dict:
    """Peak and left/right ``drop_db`` crossings.

    ``definition='relative'``: level = peak * 10^(-drop/10).
    ``definition='absolute'``: level = 10^(-drop/10).
    """
    wl = np.asarray(wl_nm, dtype=float)
    e = np.asarray(eta, dtype=float)
    peak = float(np.max(e))
    lam_peak = float(wl[int(np.argmax(e))])
    if definition == "relative":
        level = peak * 10 ** (-drop_db / 10.0)
    elif definition == "absolute":
        level = 10 ** (-drop_db / 10.0)
    else:
        raise ValueError(definition)
    left = _crossing(wl, e, level, "left")
    right = _crossing(wl, e, level, "right")
    return {
        "definition": definition,
        "peak": peak,
        "peak_dB": 10 * np.log10(peak) if peak > 0 else float("nan"),
        "lambda_peak_nm": lam_peak,
        "level": level,
        "lambda_left_nm": left,
        "lambda_right_nm": right,
        "bandwidth_nm": (right - left) if (left is not None and right is not None) else None,
    }
