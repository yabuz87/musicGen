from __future__ import annotations

from typing import Mapping, Optional, Sequence

from core_types import BeatState
from vocab import DEFAULT_VOCABULARIES, MeterToken

# ---------------------------------------------------------------------
# Rhythm / BeatState scoring weights (GTTM-inspired "energies")
# ---------------------------------------------------------------------

# Prefer keeping meter stable across adjacent beats.
W_METER_STABILITY = 1.0

# Prefer grooves to persist for some time (reduces jitter).
W_GROOVE_CONTINUITY = 0.6

# Prefer harmonic/structural boundaries to land on strong beats.
W_BOUNDARY_ON_STRONG = 0.9

# Penalize boundaries that occur on weak beats (simple syncopation penalty).
W_BOUNDARY_ON_WEAK_PENALTY = 0.7

# Small preference for being on strong beats even without a boundary
# (helps avoid too many events clustered on weak beats).
W_STRONG_BEAT_BIAS = 0.15

# Penalize illegal beat indices for the chosen meter.
W_ILLEGAL_BEAT_PENALTY = 2.0


MeterSpec = MeterToken
DEFAULT_METERS: Mapping[int, MeterSpec] = DEFAULT_VOCABULARIES.meters.id_map


def beats_per_bar(meter_id: int, meters: Optional[Mapping[int, MeterSpec]] = None) -> int:
    m = DEFAULT_METERS if meters is None else meters
    spec = m.get(meter_id)
    if spec is None:
        # Fallback: treat unknown meters as 4/4 rather than exploding.
        return 4
    return max(1, int(spec.beats_per_bar))


def is_strong_beat(beat_in_bar: int, bpb: int) -> bool:
    """Heuristic strong beat detector.

    - Beat 0 is always strong (downbeat).
    - In even meters, beat bpb/2 is also a secondary strong beat.
    - In odd meters, we treat beat 0 as strong and the rest as weak.
      (Later you can refine with a groove/meter-specific accent model.)
    """
    if beat_in_bar == 0:
        return True
    if bpb % 2 == 0 and beat_in_bar == (bpb // 2):
        return True
    return False


def illegal_beat_penalty(st: BeatState, meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    bpb = beats_per_bar(st.meter_id, meters)
    if st.beat_in_bar < 0 or st.beat_in_bar >= bpb:
        return -W_ILLEGAL_BEAT_PENALTY
    return 0.0


def strong_beat_bias(st: BeatState, meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    bpb = beats_per_bar(st.meter_id, meters)
    return W_STRONG_BEAT_BIAS if is_strong_beat(st.beat_in_bar, bpb) else 0.0


def boundary_score(st: BeatState, meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    """Score a boundary placement for a single beat.

    boundary_lvl is treated as an intensity: 0 = no boundary, higher = stronger.
    """
    if st.boundary_lvl <= 0:
        return 0.0

    bpb = beats_per_bar(st.meter_id, meters)
    if is_strong_beat(st.beat_in_bar, bpb):
        return W_BOUNDARY_ON_STRONG * float(st.boundary_lvl)
    return -W_BOUNDARY_ON_WEAK_PENALTY * float(st.boundary_lvl)


def transition_score(a: BeatState, b: BeatState, meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    """Score the transition a -> b.

    This is meant to be a light-weight analogue to gttm_features.transition_score:
    small bonuses for continuity, penalties for pathological jumps.
    """
    s = 0.0

    # Meter stability.
    if a.meter_id == b.meter_id:
        s += W_METER_STABILITY
    else:
        s -= W_METER_STABILITY

    # Groove continuity.
    if a.groove_id == b.groove_id:
        s += W_GROOVE_CONTINUITY
    else:
        s -= (W_GROOVE_CONTINUITY * 0.5)

    # Prefer boundaries in b to be well-placed.
    s += boundary_score(b, meters)

    # Small strong-beat bias for b.
    s += strong_beat_bias(b, meters)

    # Penalize illegal beat indices for b (and a little for a too).
    s += illegal_beat_penalty(b, meters) + 0.25 * illegal_beat_penalty(a, meters)

    return s


def local_window_score(prev: Optional[BeatState], curr: BeatState, next_: Optional[BeatState],
                       meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    """A tiny 3-beat window score.

    Rewards plausible "grouping" behavior:
    - If curr is a boundary, it's better when either prev is not a boundary
      (a boundary onset) or next starts a new bar (downbeat).
    """
    if prev is None or next_ is None:
        return 0.0

    if curr.boundary_lvl <= 0:
        return 0.0

    # Bonus if boundary is an onset (previous beat had no boundary).
    onset_bonus = 0.25 if prev.boundary_lvl <= 0 else 0.0

    # Bonus if after boundary we land on a strong beat (often a new group).
    next_bpb = beats_per_bar(next_.meter_id, meters)
    next_strong = 0.35 if is_strong_beat(next_.beat_in_bar, next_bpb) else 0.0

    return (onset_bonus + next_strong) * float(curr.boundary_lvl)


def sequence_score(seq: Sequence[BeatState], meters: Optional[Mapping[int, MeterSpec]] = None) -> float:
    if len(seq) < 2:
        return 0.0

    total = sum(transition_score(seq[i], seq[i + 1], meters) for i in range(len(seq) - 1))
    total += sum(
        local_window_score(seq[i - 1], seq[i], seq[i + 1], meters)
        for i in range(1, len(seq) - 1)
    )
    return float(total)
