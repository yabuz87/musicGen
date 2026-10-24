import unittest

from aimusic.core.config import PlanConfig, SectioningStrategy, StyleConfig
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import build_default_vocabularies
from aimusic.planning.plans import (
    MethodARunConfig,
    PlanningSection,
    build_executable_section_timeline,
    run_method_a,
)
from aimusic.planning.sections import SectionPlanError


class TestReq04ProductionValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.style = StyleConfig(allowed_meters=("4/4",))
        self.vocabularies = build_default_vocabularies(self.style)

    def test_run_method_a_rejects_authored_off_bar_boundary(self) -> None:
        sections = (
            PlanningSection("A", 0, 6, 2),
            PlanningSection("B", 6, 16, 3),
        )
        config = MethodARunConfig(
            total_beats=16,
            style_config=self.style,
            sections=sections,
        )

        with self.assertRaisesRegex(SectionPlanError, "does not fall on a downbeat"):
            run_method_a(config, key=RNGKey(seed=11))

    def test_generated_sections_are_snapped_to_downbeats(self) -> None:
        config = MethodARunConfig(
            total_beats=10,
            style_config=self.style,
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("A", "B"),
            ),
        )

        timeline, messages = build_executable_section_timeline(
            config,
            self.vocabularies,
        )

        self.assertEqual(
            tuple((section.start_time, section.end_time) for section in timeline.sections),
            ((0, 4), (4, 10)),
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("snapped from 5 to 4", messages[0])


if __name__ == "__main__":
    unittest.main()
