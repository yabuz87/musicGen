from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Iterator, Sequence, Tuple


ExpressiveControls = Tuple[float, ...]


def _require_int(name: str, value: int, *, minimum: int | None = None) -> None:
    """Validate that a field is an integer and optionally above a floor."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float | None = None) -> None:
    """Validate that a field is a finite real number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


@dataclass(frozen=True)
class BeatState:
    """Canonical beat-level structural state.

    The project README defines:
      St = (meter_id, beat_in_bar, boundary_lvl, key_id, chord_id, role_id, head_id, groove_id)
    """

    meter_id: int
    beat_in_bar: int
    boundary_lvl: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int

    def __post_init__(self) -> None:
        for name in (
            "meter_id",
            "beat_in_bar",
            "boundary_lvl",
            "key_id",
            "chord_id",
            "role_id",
            "head_id",
            "groove_id",
        ):
            _require_int(name, getattr(self, name), minimum=0)


@dataclass(frozen=True)
class NoteEvent:
    """Canonical score-level symbolic note event.

    Matches the design spec fields:
      (ton, toff, h, v, e, track)
    """

    ton: int
    toff: int
    h: int
    v: float
    e: ExpressiveControls = ()
    track: str = "default"

    def __post_init__(self) -> None:
        _require_int("ton", self.ton, minimum=0)
        _require_int("toff", self.toff, minimum=0)
        if self.toff <= self.ton:
            raise ValueError("toff must be > ton.")
        _require_int("h", self.h)
        _require_real("v", self.v, minimum=0.0)
        if self.v > 1.0:
            raise ValueError("v must be <= 1.0.")

        expressive_controls = tuple(self.e)
        for idx, value in enumerate(expressive_controls):
            _require_real(f"e[{idx}]", value)
        object.__setattr__(self, "e", expressive_controls)

        if not isinstance(self.track, str) or not self.track.strip():
            raise ValueError("track must be a non-empty string.")


@dataclass(frozen=True)
class Score:
    """Immutable symbolic score represented as note events."""

    note_events: Tuple[NoteEvent, ...] = ()
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0

    def __post_init__(self) -> None:
        note_events = tuple(self.note_events)
        if any(not isinstance(event, NoteEvent) for event in note_events):
            raise TypeError("note_events must contain only NoteEvent instances.")
        object.__setattr__(self, "note_events", note_events)

        _require_int("ticks_per_beat", self.ticks_per_beat, minimum=1)
        _require_real("tempo_bpm", self.tempo_bpm, minimum=0.0)
        if self.tempo_bpm == 0.0:
            raise ValueError("tempo_bpm must be > 0.")

    def __iter__(self) -> Iterator[NoteEvent]:
        return iter(self.note_events)

    def __len__(self) -> int:
        return len(self.note_events)


@dataclass(frozen=True)
class Layer:
    """Immutable collection of candidate BeatStates at a given beat index."""

    time_index: int
    states: Tuple[BeatState, ...]

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)

        states = tuple(self.states)
        if any(not isinstance(state, BeatState) for state in states):
            raise TypeError("states must contain only BeatState instances.")
        if len(states) != len(set(states)):
            raise ValueError("Layer states must be unique.")
        object.__setattr__(self, "states", states)

    def __iter__(self) -> Iterator[BeatState]:
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)


@dataclass(frozen=True)
class Edge:
    """Immutable transition edge between two BeatStates."""

    time_index: int
    source: BeatState
    target: BeatState
    log_weight: float

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.source, BeatState):
            raise TypeError("source must be a BeatState.")
        if not isinstance(self.target, BeatState):
            raise TypeError("target must be a BeatState.")
        _require_real("log_weight", self.log_weight)


@dataclass(frozen=True)
class EndpointDistribution:
    """Normalized probability distribution over a specific graph layer."""

    layer: Layer
    probabilities: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layer, Layer):
            raise TypeError("layer must be a Layer.")
        if len(self.layer) == 0:
            raise ValueError("EndpointDistribution layer must not be empty.")

        probabilities = tuple(self.probabilities)
        if len(probabilities) != len(self.layer):
            raise ValueError("probabilities must align 1:1 with the layer states.")
        for idx, prob in enumerate(probabilities):
            _require_real(f"probabilities[{idx}]", prob, minimum=0.0)

        total = float(sum(probabilities))
        if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("EndpointDistribution probabilities must sum to 1.0.")

        object.__setattr__(self, "probabilities", probabilities)

    def probability_of(self, state: BeatState) -> float:
        """Return the probability mass of *state*, or 0.0 if absent."""
        for candidate, probability in zip(self.layer.states, self.probabilities):
            if candidate == state:
                return probability
        return 0.0

