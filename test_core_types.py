import unittest
from dataclasses import FrozenInstanceError

from core_types import BeatState, Edge, EndpointDistribution, Layer, NoteEvent, Score
from vocab import DEFAULT_VOCABULARIES


class TestBeatState(unittest.TestCase):
    def test_beat_state_is_hashable_and_value_equal(self):
        a = BeatState(0, 1, 2, 3, 4, 5, 6, 7)
        b = BeatState(0, 1, 2, 3, 4, 5, 6, 7)

        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_beat_state_rejects_negative_tokens(self):
        with self.assertRaises(ValueError):
            BeatState(-1, 0, 0, 0, 0, 0, 0, 0)

    def test_beat_state_serializes_with_optional_vocab_labels(self):
        state = BeatState(0, 1, 2, 0, 0, 0, 1, 0)

        raw = state.to_dict()
        labeled = state.to_dict(DEFAULT_VOCABULARIES)
        pretty = state.pretty(DEFAULT_VOCABULARIES)

        self.assertEqual(raw["meter_id"], 0)
        self.assertEqual(labeled["meter_label"], "4/4")
        self.assertEqual(labeled["beat_label"], "beat_2")
        self.assertEqual(labeled["boundary_label"], "phrase")
        self.assertIn("meter=4/4[0]", pretty)
        self.assertIn("head=root[1]", pretty)


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

    def test_note_event_serializes_to_json_friendly_mapping(self):
        event = NoteEvent(ton=0, toff=120, h=7, v=0.75, e=(0.1, 0.2), track="lead")

        data = event.to_dict()
        pretty = event.pretty()

        self.assertEqual(data["duration_ticks"], 120)
        self.assertEqual(data["e"], [0.1, 0.2])
        self.assertIn("track=lead", pretty)
        self.assertIn("ticks=0->120", pretty)


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

    def test_score_serialization_and_pretty_summary(self):
        note_a = NoteEvent(0, 120, 0, 0.7, track="bass")
        note_b = NoteEvent(120, 240, 4, 0.8, track="lead")
        score = Score(note_events=(note_a, note_b), ticks_per_beat=240, tempo_bpm=110.0)

        data = score.to_dict()
        pretty = score.pretty(max_events=1)

        self.assertEqual(data["event_count"], 2)
        self.assertEqual(data["track_event_counts"], {"bass": 1, "lead": 1})
        self.assertEqual(len(data["note_events"]), 2)
        self.assertIn("events=2", pretty)
        self.assertIn("tracks={bass:1, lead:1}", pretty)
        self.assertIn("...", pretty)


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

    def test_layer_serialization_and_pretty_preview(self):
        states = (
            BeatState(0, 0, 0, 0, 0, 0, 0, 0),
            BeatState(0, 1, 1, 0, 0, 0, 1, 0),
        )
        layer = Layer(time_index=3, states=states)

        data = layer.to_dict(DEFAULT_VOCABULARIES)
        pretty = layer.pretty(DEFAULT_VOCABULARIES, max_states=1)

        self.assertEqual(data["time_index"], 3)
        self.assertEqual(data["size"], 2)
        self.assertEqual(data["states"][1]["boundary_label"], "local")
        self.assertIn("Layer(t=3, size=2", pretty)
        self.assertIn("...", pretty)


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

    def test_edge_serialization_and_pretty_output(self):
        source = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        target = BeatState(0, 1, 1, 0, 0, 0, 1, 0)
        edge = Edge(time_index=0, source=source, target=target, log_weight=-1.25)

        data = edge.to_dict(DEFAULT_VOCABULARIES)
        pretty = edge.pretty(DEFAULT_VOCABULARIES)

        self.assertEqual(data["time_index"], 0)
        self.assertEqual(data["source"]["meter_label"], "4/4")
        self.assertEqual(data["target"]["head_label"], "root")
        self.assertIn("log_weight=-1.250", pretty)
        self.assertIn("source=BeatState(", pretty)


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

    def test_endpoint_distribution_serialization_and_pretty_output(self):
        states = (
            BeatState(0, 0, 0, 0, 0, 0, 0, 0),
            BeatState(0, 1, 1, 0, 0, 0, 1, 0),
        )
        endpoint = EndpointDistribution(
            layer=Layer(time_index=8, states=states),
            probabilities=(0.4, 0.6),
        )

        data = endpoint.to_dict(DEFAULT_VOCABULARIES)
        pretty = endpoint.pretty(DEFAULT_VOCABULARIES, max_states=1)

        self.assertEqual(data["time_index"], 8)
        self.assertEqual(data["support"][1]["state"]["head_label"], "root")
        self.assertEqual(data["support"][0]["probability"], 0.4)
        self.assertIn("EndpointDistribution(t=8, size=2", pretty)
        self.assertIn("@0.400", pretty)


class TestCoreTypeImmutability(unittest.TestCase):
    def test_core_types_are_frozen(self):
        state = BeatState(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(FrozenInstanceError):
            state.beat_in_bar = 1


if __name__ == "__main__":
    unittest.main()
