import unittest
from dataclasses import FrozenInstanceError

from core_types import BeatState, Edge, EndpointDistribution, Layer, NoteEvent, Score


class TestBeatState(unittest.TestCase):
    def test_beat_state_is_hashable_and_value_equal(self):
        a = BeatState(0, 1, 2, 3, 4, 5, 6, 7)
        b = BeatState(0, 1, 2, 3, 4, 5, 6, 7)

        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_beat_state_rejects_negative_tokens(self):
        with self.assertRaises(ValueError):
            BeatState(-1, 0, 0, 0, 0, 0, 0, 0)


class TestNoteEvent(unittest.TestCase):
    def test_note_event_coerces_expression_controls(self):
        event = NoteEvent(ton=0, toff=120, h=7, v=0.75, e=[0.1, 0.2], track="lead")

        self.assertEqual(event.e, (0.1, 0.2))
        self.assertEqual(event.track, "lead")

    def test_note_event_rejects_invalid_duration_velocity_and_track(self):
        with self.assertRaises(ValueError):
            NoteEvent(ton=10, toff=10, h=0, v=0.5)
        with self.assertRaises(ValueError):
            NoteEvent(ton=0, toff=10, h=0, v=1.1)
        with self.assertRaises(ValueError):
            NoteEvent(ton=0, toff=10, h=0, v=0.5, track="")


class TestScore(unittest.TestCase):
    def test_score_coerces_note_events_to_tuple_and_is_iterable(self):
        note_a = NoteEvent(0, 120, 0, 0.7, track="bass")
        note_b = NoteEvent(120, 240, 4, 0.8, track="lead")
        score = Score(note_events=[note_a, note_b], ticks_per_beat=240, tempo_bpm=110.0)

        self.assertEqual(len(score), 2)
        self.assertEqual(tuple(score), (note_a, note_b))
        self.assertEqual(score.note_events, (note_a, note_b))

    def test_score_rejects_invalid_note_event_payloads(self):
        with self.assertRaises(TypeError):
            Score(note_events=(object(),))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Score(ticks_per_beat=0)


class TestLayer(unittest.TestCase):
    def test_layer_requires_unique_states(self):
        state = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            Layer(time_index=0, states=(state, state))

    def test_layer_is_iterable(self):
        states = (
            BeatState(0, 0, 0, 0, 0, 0, 0, 0),
            BeatState(0, 1, 0, 0, 0, 0, 0, 0),
        )
        layer = Layer(time_index=3, states=states)

        self.assertEqual(len(layer), 2)
        self.assertEqual(tuple(layer), states)


class TestEdge(unittest.TestCase):
    def test_edge_accepts_finite_log_weight(self):
        source = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        target = BeatState(0, 1, 0, 0, 0, 0, 0, 1)
        edge = Edge(time_index=0, source=source, target=target, log_weight=-1.25)

        self.assertEqual(edge.source, source)
        self.assertEqual(edge.target, target)

    def test_edge_rejects_non_state_endpoints(self):
        source = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(TypeError):
            Edge(time_index=0, source=source, target=object(), log_weight=0.0)  # type: ignore[arg-type]


class TestEndpointDistribution(unittest.TestCase):
    def test_endpoint_distribution_validates_probabilities(self):
        states = (
            BeatState(0, 0, 0, 0, 0, 0, 0, 0),
            BeatState(0, 1, 0, 0, 0, 0, 0, 1),
        )
        layer = Layer(time_index=8, states=states)
        endpoint = EndpointDistribution(layer=layer, probabilities=[0.4, 0.6])

        self.assertEqual(endpoint.probability_of(states[0]), 0.4)
        self.assertEqual(endpoint.probability_of(states[1]), 0.6)
        self.assertEqual(endpoint.probability_of(BeatState(0, 2, 0, 0, 0, 0, 0, 2)), 0.0)

    def test_endpoint_distribution_rejects_empty_or_non_normalized_support(self):
        empty_layer = Layer(time_index=0, states=())
        state = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        layer = Layer(time_index=0, states=(state,))

        with self.assertRaises(ValueError):
            EndpointDistribution(layer=empty_layer, probabilities=())
        with self.assertRaises(ValueError):
            EndpointDistribution(layer=layer, probabilities=(0.5,))


class TestCoreTypeImmutability(unittest.TestCase):
    def test_core_types_are_frozen(self):
        state = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(FrozenInstanceError):
            state.beat_in_bar = 1


if __name__ == "__main__":
    unittest.main()
