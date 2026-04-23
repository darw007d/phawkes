"""Per-regime Hawkes calibration — one set of kernels per regime label.

Motivation: in financial contagion, pooling event streams across different
market regimes (calm / elevated / crisis) causes spurious criticality — the
MLE misreads regime shifts as endogenous self-excitation and pushes the
branching ratio toward 1 artificially. Fitting a separate kernel per regime
restores identifiability.

Inputs:
- per-dim event streams on an observation window [0, T]
- regime_segments: list of (t_start, t_end, label) tuples partitioning [0, T]

For each label, we slice each dim's event stream to times inside the
relevant segment(s), shift to [0, L] local time, and call HawkesExpMLE.

Returns a dict mapping label → fitted HawkesExpMLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phawkes.exp_mle import HawkesExpMLE


@dataclass
class RegimeSegment:
    """A contiguous time interval labelled with a regime identifier."""

    t_start: float
    t_end: float
    label: str

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end must be > t_start, got {self.t_start} .. {self.t_end}")


def fit_per_regime(
    events: Sequence[np.ndarray],
    segments: Sequence[RegimeSegment],
    decay: float,
    penalty_l2: float = 0.0,
    min_events_per_regime: int = 50,
) -> dict[str, HawkesExpMLE]:
    """Fit a separate Hawkes MLE per regime label.

    Args:
        events: length-d list of sorted event arrays (global times).
        segments: non-overlapping intervals labelling the observation window
            by regime. Multiple segments may share a label (e.g. crisis
            recurs) — their local streams are concatenated with proper
            time-shift and the effective observation window is the sum of
            segment lengths.
        decay: exponential kernel decay beta (shared across dims, v0.1).
        penalty_l2: ridge on log_alpha (passed to HawkesExpMLE).
        min_events_per_regime: skip labels with fewer total events than
            this — fitting will be meaningless.

    Returns:
        {label: fitted HawkesExpMLE}. Labels with too few events are omitted
        and the skipped-labels list is attached as attribute `skipped_`.
    """
    if decay <= 0:
        raise ValueError(f"decay must be positive, got {decay}")
    if not segments:
        raise ValueError("need at least one regime segment")

    # group segments by label
    by_label: dict[str, list[RegimeSegment]] = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)

    d = len(events)
    out: dict[str, HawkesExpMLE] = {}
    skipped: list[tuple[str, int]] = []

    for label, segs in by_label.items():
        # concatenate local event streams, shifting each segment to start at
        # the current accumulated local-time cursor
        per_dim_local: list[list[float]] = [[] for _ in range(d)]
        cursor = 0.0
        for seg in segs:
            seg_len = seg.t_end - seg.t_start
            for i, ts in enumerate(events):
                mask = (ts >= seg.t_start) & (ts < seg.t_end)
                local = ts[mask] - seg.t_start + cursor
                per_dim_local[i].extend(local.tolist())
            cursor += seg_len

        local_events = [np.sort(np.asarray(x, dtype=np.float64)) for x in per_dim_local]
        total_evts = sum(t.size for t in local_events)
        if total_evts < min_events_per_regime:
            skipped.append((label, total_evts))
            continue

        learner = HawkesExpMLE(decay=decay, penalty_l2=penalty_l2).fit(local_events, end_time=cursor)
        out[label] = learner

    # attach diagnostic
    out_any_learner = next(iter(out.values()), None)
    if out_any_learner is not None:
        out_any_learner.__dict__.setdefault("skipped_", skipped)
    return out
