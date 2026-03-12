from dataclasses import dataclass
from enum import Enum


class MicrotonalRendering(Enum):
    """How to render non-12-EDO pitches to MIDI."""
    MPE = "mpe"  # Per-note pitch bends
    MTS = "mts"  # MIDI Tuning Standard


@dataclass(frozen=True)
class EDOConfig:
    """
    Configuration for an Equal Division of the Octave (EDO) system.

    Attributes:
        n: The number of equal divisions of the octave.
        base_tuning: The MIDI note number for pitch 0 (e.g., 60 for middle C).
        microtonal_rendering_method: The method for rendering non-12-EDO pitches.
        pitch_bend_range: The range of pitch bends in semitones.
    """
    n: int
    base_tuning: float = 60.0
    microtonal_rendering_method: MicrotonalRendering = MicrotonalRendering.MPE
    pitch_bend_range: int = 48
