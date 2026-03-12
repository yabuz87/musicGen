from edo import EDO


class TonalSystem:
    """
    Defines a tonal system for a given EDO.

    This class would be responsible for things like:
    - Defining scales and chords.
    - Calculating tonal distances.
    - Providing chord templates.
    """

    def __init__(self, edo: EDO):
        self.edo = edo

    def __repr__(self) -> str:
        return f"TonalSystem(edo={self.edo})"
