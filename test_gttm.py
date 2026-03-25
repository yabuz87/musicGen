import unittest
from gttm_features import ChordState, sequence_score, pc, tonal_distance, is_fifth_down

class TestGTTMRequirements(unittest.TestCase):
    
    def test_edo_genericity(self):
        """
        Check if the system handles 19-EDO correctly.
        In 19-EDO, a 'perfect fifth' is 11 steps, not 7.
        """
        self.assertEqual(pc(19, edo=19), 0)
        self.assertTrue(is_fifth_down(11, 0, edo=19), "Should identify 19-EDO fifth")

    def test_energy_summation(self):
        """
        Check if the sequence_score correctly aggregates multiple rules.
        """
        c1 = ChordState(root_pc=2, quality="m7", duration=1.0, bar_position=0.0) # ii
        c2 = ChordState(root_pc=7, quality="7", duration=1.0, bar_position=0.2)  # V
        c3 = ChordState(root_pc=0, quality="maj7", duration=2.0, bar_position=0.0)# I
        
        seq = [c1, c2, c3]
        score = sequence_score(seq, edo=12)
        self.assertGreater(score, 0, "Energy score should be a positive summation for a valid ii-V-I")

    def test_tonal_distance_metric(self):
        """
        Check if a distance metric exists and correctly penalizes distant keys.
        """
        dist_near = tonal_distance(0, 7, edo=12)
        dist_far = tonal_distance(0, 6, edo=12)
        self.assertLess(dist_near, dist_far, "Tonal distance should reflect the Circle of Fifths")

if __name__ == "__main__":
    unittest.main()