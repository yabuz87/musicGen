from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    NeuralPriorConfig,
    PlanConfig,
    PlanMethod,
    PriorWeights,
    SBConfig,
    SectionStyleWeights,
    SectioningStrategy,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer, Score
from aimusic.core.rng import RNGKey, allocate_named_keys, random_unit
from aimusic.core.vocab import TonalContext, Vocabularies, build_tonal_context
from aimusic.decode import decode_path_to_score
from aimusic.planning.graph import EdgeScoreDiagnostics, SparseGraph, build_sparse_graph
from aimusic.planning.sb import (
    SBProblem,
    SBSolution,
    SampledBridgePath,
    SolvedBridge,
    build_sb_problem,
    map_bridge_path,
    sample_bridge_path,
    solve_sb,
)
from aimusic.scoring.priors import NullPrior, Prior
from aimusic.theory.tonal import get_fifth_steps


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _state_sort_key(state: BeatState) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        state.meter_id,
        state.beat_in_bar,
        state.boundary_lvl,
        state.key_id,
        state.chord_id,
        state.role_id,
        state.head_id,
        state.groove_id,
    )


from aimusic.planning.sections import (
    PlanningSection,
    SectionPlanError,
    SectionTimeline,
)


def get_section_at_time(
    sections: Sequence[PlanningSection] | SectionTimeline, t: int
) -> Optional[PlanningSection]:
    """Find the PlanningSection governing state index t.

    .. deprecated::
        Use :class:`SectionTimeline` and ``timeline.at_state(k)`` instead.
        This is a STATE lookup (beat state s_t belongs to section(t)).
    """
    import warnings
    warnings.warn(
        "get_section_at_time is deprecated; use SectionTimeline.at_state instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(sections, SectionTimeline):
        try:
            return sections.at_state(t)
        except (IndexError, SectionPlanError):
            return None
    try:
        total_beats = max(s.end_time for s in sections) if sections else 0
        timeline = SectionTimeline(tuple(sections), total_beats=total_beats)
        return timeline.at_state(t)
    except Exception:
        for section in sections:
            if section.start_time <= t < section.end_time:
                return section
        return None



@dataclass(frozen=True)
class MethodARunConfig:
    """Pure run configuration bundle for EPIC 6 Method A orchestration."""

    total_beats: int
    seed: int = 0
    use_sampling: bool = False
    style_config: StyleConfig = field(default_factory=StyleConfig)
    prior_weights: PriorWeights = field(default_factory=PriorWeights)
    sb_config: Optional[SBConfig] = None
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    plan_config: PlanConfig = field(default_factory=PlanConfig)
    neural_prior_config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    edo: int = 12
    sections: Optional[Tuple[PlanningSection, ...]] = None

    def __post_init__(self) -> None:
        _require_int("total_beats", self.total_beats, minimum=1)
        _require_int("seed", self.seed, minimum=0)
        if not isinstance(self.use_sampling, bool):
            raise TypeError("use_sampling must be a bool.")
        if self.plan_config.method is not PlanMethod.METHOD_A:
            raise ValueError("MethodARunConfig requires plan_config.method == METHOD_A.")
        _require_int("edo", self.edo, minimum=1)
        if self.sb_config is not None and self.sb_config.horizon_t != self.total_beats:
            raise ValueError("sb_config.horizon_t must equal total_beats for Method A runs.")
        if (
            self.plan_config.sectioning_strategy is SectioningStrategy.SECTION_WISE
            and len(self.plan_config.section_names) > self.total_beats
        ):
            raise ValueError(
                "SECTION_WISE planning requires total_beats >= len(section_names)."
            )
        if self.sections is not None:
            secs = tuple(self.sections)
            if not secs:
                raise ValueError("sections tuple cannot be empty.")
            if any(not isinstance(s, PlanningSection) for s in secs):
                raise TypeError("sections must contain only PlanningSection instances.")
            object.__setattr__(self, "sections", secs)


@dataclass(frozen=True)
class EndpointChoice:
    """Explicit chosen endpoint state plus provenance within a candidate distribution."""

    state: BeatState
    source_distribution: EndpointDistribution
    selected_index: int
    selected_probability: float
    selection_mode: str

    def __post_init__(self) -> None:
        if self.state not in self.source_distribution.layer.states:
            raise ValueError("state must belong to source_distribution.layer.")
        _require_int("selected_index", self.selected_index, minimum=0)
        if self.selected_index >= len(self.source_distribution.layer.states):
            raise ValueError("selected_index must be within source_distribution support.")
        _require_real("selected_probability", self.selected_probability)
        if not isinstance(self.selection_mode, str) or not self.selection_mode.strip():
            raise ValueError("selection_mode must be a non-empty string.")


@dataclass(frozen=True)
class MethodAEndpoints:
    """Endpoint distributions and section metadata for a Method A run."""

    pi0: EndpointDistribution
    piT: EndpointDistribution
    start_choice: EndpointChoice
    end_choice: EndpointChoice
    sections: Tuple[PlanningSection, ...]


def _parse_beat_state(val: Any) -> BeatState:
    if isinstance(val, BeatState):
        return val
    if isinstance(val, dict):
        return BeatState(
            meter_id=int(val["meter_id"]),
            beat_in_bar=int(val["beat_in_bar"]),
            boundary_lvl=int(val["boundary_lvl"]),
            key_id=int(val["key_id"]),
            chord_id=int(val["chord_id"]),
            role_id=int(val["role_id"]),
            head_id=int(val["head_id"]),
            groove_id=int(val["groove_id"]),
        )
    raise TypeError(f"Cannot parse BeatState from {val!r}")


@dataclass(frozen=True)
class TransitionDiagnostic:
    """Per-transition diagnostic metrics for a selected path."""

    time_index: int
    arrival_time: int
    section_name: str
    section_index: int
    is_section_start: bool
    source_state: BeatState
    target_state: BeatState
    data_logp: float
    data_contribution: float
    gttm_raw_features: Mapping[str, float]
    gttm_weighted_features: Mapping[str, float]
    gttm_family_breakdown: Mapping[str, float]
    gttm_contribution: float
    target_tension: float
    realized_tension: float
    tension_deviation: float
    target_tension_energy: float
    target_tension_contribution: float
    section_style_components: Mapping[str, float]
    section_style_energy: float
    section_style_contribution: float
    final_log_weight: float
    right_context_states: Tuple[BeatState, ...] = ()

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        _require_int("arrival_time", self.arrival_time, minimum=0)
        if not isinstance(self.section_name, str) or not self.section_name.strip():
            raise ValueError("section_name must be a non-empty string.")
        _require_int("section_index", self.section_index, minimum=0)
        if not isinstance(self.is_section_start, bool):
            raise TypeError("is_section_start must be a bool.")
        if not isinstance(self.source_state, BeatState):
            raise TypeError("source_state must be a BeatState.")
        if not isinstance(self.target_state, BeatState):
            raise TypeError("target_state must be a BeatState.")
        for name in (
            "data_logp",
            "data_contribution",
            "gttm_contribution",
            "target_tension",
            "realized_tension",
            "tension_deviation",
            "target_tension_energy",
            "target_tension_contribution",
            "section_style_energy",
            "section_style_contribution",
            "final_log_weight",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val):
                raise ValueError(f"{name} must be a finite real number, got {val!r}")
        if not isinstance(self.gttm_raw_features, (dict, Mapping)):
            raise TypeError("gttm_raw_features must be a Mapping.")
        if not isinstance(self.gttm_weighted_features, (dict, Mapping)):
            raise TypeError("gttm_weighted_features must be a Mapping.")
        if not isinstance(self.gttm_family_breakdown, (dict, Mapping)):
            raise TypeError("gttm_family_breakdown must be a Mapping.")
        if not isinstance(self.section_style_components, (dict, Mapping)):
            raise TypeError("section_style_components must be a Mapping.")
        if not isinstance(self.right_context_states, tuple):
            object.__setattr__(self, "right_context_states", tuple(self.right_context_states))
        for state in self.right_context_states:
            if not isinstance(state, BeatState):
                raise TypeError("All items in right_context_states must be BeatState instances.")

    def verify_decomposition(self, tol: float = 1e-9) -> bool:
        """Check that final_log_weight equals data + gttm - tension - style contributions within tol."""
        if not isinstance(tol, (int, float)) or tol <= 0.0 or not np.isfinite(tol):
            raise ValueError(f"Tolerance tol must be a positive finite real number, got {tol!r}")
        expected = (
            self.data_contribution
            + self.gttm_contribution
            - self.target_tension_contribution
            - self.section_style_contribution
        )
        diff = abs(self.final_log_weight - expected)
        if diff > tol:
            raise ValueError(
                f"Transition {self.time_index} (arrival beat {self.arrival_time}, section '{self.section_name}') "
                f"decomposition failure: final_log_weight={self.final_log_weight:.10f} != "
                f"sum_of_contributions={expected:.10f} (diff={diff:.2e} > tol={tol:.2e}). "
                f"Components: data_contribution={self.data_contribution:.6f}, "
                f"gttm_contribution={self.gttm_contribution:.6f}, "
                f"-target_tension_contribution={-self.target_tension_contribution:.6f}, "
                f"-section_style_contribution={-self.section_style_contribution:.6f}"
            )
        return True

    def to_dict(self, vocabularies: Optional[Vocabularies] = None) -> Dict[str, Any]:
        """Serialize transition diagnostic record to JSON-compatible dictionary."""
        return {
            "time_index": self.time_index,
            "arrival_time": self.arrival_time,
            "section_name": self.section_name,
            "section_index": self.section_index,
            "is_section_start": self.is_section_start,
            "source_state": self.source_state.to_dict(vocabularies),
            "target_state": self.target_state.to_dict(vocabularies),
            "data_logp": self.data_logp,
            "data_contribution": self.data_contribution,
            "gttm_raw_features": dict(self.gttm_raw_features),
            "gttm_weighted_features": dict(self.gttm_weighted_features),
            "gttm_family_breakdown": dict(self.gttm_family_breakdown),
            "gttm_contribution": self.gttm_contribution,
            "target_tension": self.target_tension,
            "realized_tension": self.realized_tension,
            "tension_deviation": self.tension_deviation,
            "target_tension_energy": self.target_tension_energy,
            "target_tension_contribution": self.target_tension_contribution,
            "section_style_components": dict(self.section_style_components),
            "section_style_energy": self.section_style_energy,
            "section_style_contribution": self.section_style_contribution,
            "final_log_weight": self.final_log_weight,
            "right_context_states": [s.to_dict(vocabularies) for s in self.right_context_states],
            # Legacy fields for backward compatibility
            "gttm_energy": self.gttm_energy,
            "target_tension_penalty": self.target_tension_contribution,
            "section_style_penalty": self.section_style_contribution,
            "section_style_breakdown": dict(self.section_style_components),
            "total_log_weight": self.final_log_weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionDiagnostic":
        """Deserialize transition diagnostic record, supporting new and legacy schemas."""
        t = int(data["time_index"])
        arrival_time = int(data.get("arrival_time", t + 1))
        section_name = str(data.get("section_name", "A"))
        section_index = int(data.get("section_index", 0))
        is_section_start = bool(data.get("is_section_start", False))
        source_state = _parse_beat_state(data["source_state"])
        target_state = _parse_beat_state(data["target_state"])

        data_logp = float(data.get("data_logp", 0.0))
        data_contribution = float(data.get("data_contribution", data_logp))

        gttm_raw = dict(data.get("gttm_raw_features", {}))
        gttm_weighted = dict(data.get("gttm_weighted_features", {}))
        gttm_family = dict(data.get("gttm_family_breakdown", {}))

        if "gttm_contribution" in data:
            gttm_contribution = float(data["gttm_contribution"])
        elif "gttm_energy" in data:
            gttm_contribution = -float(data["gttm_energy"])
        elif gttm_weighted:
            gttm_contribution = float(sum(gttm_weighted.values()))
        else:
            gttm_contribution = 0.0

        target_tension = float(data.get("target_tension", 0.0))
        realized_tension = float(data.get("realized_tension", 0.0))
        tension_deviation = float(data.get("tension_deviation", realized_tension - target_tension))

        if "target_tension_contribution" in data:
            t_contrib = float(data["target_tension_contribution"])
            t_energy = float(data.get("target_tension_energy", t_contrib))
        else:
            t_contrib = float(data.get("target_tension_penalty", 0.0))
            t_energy = float(data.get("target_tension_energy", t_contrib))

        if "section_style_contribution" in data:
            s_contrib = float(data["section_style_contribution"])
            s_energy = float(data.get("section_style_energy", s_contrib))
        else:
            s_contrib = float(data.get("section_style_penalty", 0.0))
            s_energy = float(data.get("section_style_energy", s_contrib))

        style_components = dict(data.get("section_style_components", data.get("section_style_breakdown", {})))

        if "final_log_weight" in data:
            final_log_weight = float(data["final_log_weight"])
        elif "total_log_weight" in data:
            final_log_weight = float(data["total_log_weight"])
        else:
            final_log_weight = float(data_contribution + gttm_contribution - t_contrib - s_contrib)

        right_contexts_raw = data.get("right_context_states", ())
        right_contexts = tuple(_parse_beat_state(s) for s in right_contexts_raw)

        return cls(
            time_index=t,
            arrival_time=arrival_time,
            section_name=section_name,
            section_index=section_index,
            is_section_start=is_section_start,
            source_state=source_state,
            target_state=target_state,
            data_logp=data_logp,
            data_contribution=data_contribution,
            gttm_raw_features=gttm_raw,
            gttm_weighted_features=gttm_weighted,
            gttm_family_breakdown=gttm_family,
            gttm_contribution=gttm_contribution,
            target_tension=target_tension,
            realized_tension=realized_tension,
            tension_deviation=tension_deviation,
            target_tension_energy=t_energy,
            target_tension_contribution=t_contrib,
            section_style_components=style_components,
            section_style_energy=s_energy,
            section_style_contribution=s_contrib,
            final_log_weight=final_log_weight,
            right_context_states=right_contexts,
        )

    @property
    def total_log_weight(self) -> float:
        return self.final_log_weight

    @property
    def target_tension_penalty(self) -> float:
        return self.target_tension_contribution

    @property
    def section_style_penalty(self) -> float:
        return self.section_style_contribution

    @property
    def section_style_breakdown(self) -> Mapping[str, float]:
        return self.section_style_components

    @property
    def gttm_energy(self) -> float:
        return -float(sum(self.gttm_weighted_features.values())) if self.gttm_weighted_features else 0.0



@dataclass(frozen=True)
class MethodAPlanDiagnostics:
    """Inspectable diagnostics emitted by Method A orchestration."""

    section_tags: Tuple[str, ...]
    target_tension_arcs: Tuple[Tuple[float, ...], ...]
    chosen_start_state: BeatState
    chosen_end_state: BeatState
    endpoint_selection_mode: str
    chosen_start_probability: float
    chosen_end_probability: float
    path_mode: str
    graph_layer_sizes: Tuple[int, ...]
    bridge_iterations: int
    bridge_converged: bool
    rng_stream_ids: Tuple[str, ...]
    transition_diagnostics: Tuple[TransitionDiagnostic, ...] = ()
    timeline: Optional[SectionTimeline] = None


@dataclass(frozen=True)
class MethodAPlanResult:
    """Full output of a Method A planning pass."""

    run_config: MethodARunConfig
    tonal_context: TonalContext
    vocabularies: Vocabularies
    endpoints: MethodAEndpoints
    graph: SparseGraph
    sb_problem: SBProblem
    sb_solution: SBSolution
    bridge: SolvedBridge
    path: Tuple[BeatState, ...]
    path_score: Optional[float]
    sampled_path: Optional[SampledBridgePath]
    diagnostics: MethodAPlanDiagnostics
    path_edge_diagnostics: Tuple[EdgeScoreDiagnostics, ...] = ()
    timeline: Optional[SectionTimeline] = None


@dataclass(frozen=True)
class ExactBridgeDemoResult:
    """Compatibility wrapper for legacy bridge-demo scripts."""

    plan_result: MethodAPlanResult
    score: Score
    output_path: str


def _resolved_vocabs(
    vocabularies: Optional[Vocabularies],
    style_config: StyleConfig,
    edo: int,
) -> Vocabularies:
    return build_tonal_context(
        edo,
        style_config,
        vocabularies=vocabularies,
    ).vocabularies


def _resolved_sb_config(run_config: MethodARunConfig) -> SBConfig:
    if run_config.sb_config is not None:
        return run_config.sb_config
    return SBConfig(horizon_t=run_config.total_beats)


def _softmax(scores: Sequence[float], temperature: float) -> Tuple[float, ...]:
    logits = np.asarray(tuple(float(score) for score in scores), dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence.")
    scaled = logits / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    normalized = weights / np.sum(weights)
    return tuple(float(value) for value in normalized)


def _align_endpoint_distribution(
    endpoint: EndpointDistribution,
    layer: Layer,
) -> EndpointDistribution:
    masses = [endpoint.probability_of(state) for state in layer.states]
    total = float(sum(masses))
    if total <= 0.0:
        raise ValueError("Endpoint support vanished after graph construction.")
    return EndpointDistribution(
        layer=layer,
        probabilities=tuple(mass / total for mass in masses),
    )


def _singleton_endpoint_distribution(state: BeatState, *, time_index: int) -> EndpointDistribution:
    return EndpointDistribution(
        layer=Layer(time_index=time_index, states=(state,)),
        probabilities=(1.0,),
    )


def _sample_index_from_distribution(
    endpoint: EndpointDistribution,
    key: RNGKey,
) -> tuple[int, RNGKey]:
    threshold, next_key = random_unit(key)
    running = 0.0
    for idx, probability in enumerate(endpoint.probabilities):
        running += probability
        if threshold <= running:
            return idx, next_key
    return len(endpoint.probabilities) - 1, next_key


def _choose_endpoint_state(
    endpoint: EndpointDistribution,
    *,
    key: RNGKey,
    sample: bool,
) -> tuple[EndpointChoice, RNGKey]:
    if sample:
        selected_index, next_key = _sample_index_from_distribution(endpoint, key)
        selection_mode = "sample"
    else:
        selected_index = max(
            range(len(endpoint.probabilities)),
            key=lambda idx: (endpoint.probabilities[idx], -idx),
        )
        next_key = key
        selection_mode = "argmax"
    return (
        EndpointChoice(
            state=endpoint.layer.states[selected_index],
            source_distribution=endpoint,
            selected_index=selected_index,
            selected_probability=endpoint.probabilities[selected_index],
            selection_mode=selection_mode,
        ),
        next_key,
    )


def build_section_plan(run_config: MethodARunConfig) -> Tuple[PlanningSection, ...]:
    is_multi_meter = len(run_config.style_config.allowed_meters) > 1
    if run_config.sections is not None:
        SectionTimeline(
            run_config.sections,
            run_config.total_beats,
            edo=run_config.edo,
            multi_meter=is_multi_meter,
        )
        return run_config.sections

    plan_config = run_config.plan_config
    if plan_config.sectioning_strategy is SectioningStrategy.SINGLE_PASS:
        name = (
            plan_config.section_names[0]
            if plan_config.section_names
            else "method_a_single_pass"
        )
        sections = (
            PlanningSection(
                name=name,
                start_time=0,
                end_time=run_config.total_beats,
                boundary_level=3,
                allowed_meters=run_config.style_config.allowed_meters,
                groove_family=None,
            ),
        )
        SectionTimeline(
            sections,
            run_config.total_beats,
            edo=run_config.edo,
            multi_meter=is_multi_meter,
        )
        return sections

    section_names = plan_config.section_names
    section_count = len(section_names)
    if section_count > run_config.total_beats:
        raise ValueError(
            "SECTION_WISE planning requires total_beats >= len(section_names)."
        )
    chunk = run_config.total_beats // section_count
    remainder = run_config.total_beats % section_count
    sections_list = []
    cursor = 0
    for idx, name in enumerate(section_names):
        length = chunk + (1 if idx < remainder else 0)
        next_cursor = cursor + max(1, length)
        start_tension = min(0.9, 0.2 + (0.1 * idx))
        sections_list.append(
            PlanningSection(
                name=name,
                start_time=cursor,
                end_time=next_cursor,
                boundary_level=3 if idx == section_count - 1 else 2,
                target_tension_arc=(start_tension, 0.8, 0.25),
                allowed_meters=run_config.style_config.allowed_meters,
                groove_family=None,
            )
        )
        cursor = next_cursor
    last = sections_list[-1]
    if last.end_time != run_config.total_beats:
        sections_list[-1] = PlanningSection(
            name=last.name,
            start_time=last.start_time,
            end_time=run_config.total_beats,
            boundary_level=last.boundary_level,
            target_tension_arc=last.target_tension_arc,
            allowed_meters=last.allowed_meters,
            groove_family=None,
            target_key_id=last.target_key_id,
            key_region=last.key_region,
            target_density=last.target_density,
            preferred_roles=last.preferred_roles,
        )
    plan = tuple(sections_list)
    SectionTimeline(
        plan,
        run_config.total_beats,
        edo=run_config.edo,
        multi_meter=is_multi_meter,
    )
    return plan


def build_executable_section_timeline(
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
) -> tuple[SectionTimeline, Tuple[str, ...]]:
    """Build and validate the section timeline used by production planning.

    Authored section boundaries must already align with a single-meter plan.
    Automatically partitioned sections are snapped to downbeats. Multi-meter
    alignment remains a soft target because its bar positions depend on the
    selected path.
    """
    sections = (
        run_config.sections
        if run_config.sections is not None
        else build_section_plan(run_config)
    )
    timeline = SectionTimeline(
        sections=sections,
        total_beats=run_config.total_beats,
        edo=run_config.edo,
        multi_meter=len(run_config.style_config.allowed_meters) > 1,
    )
    validation = timeline.validate_executable(
        run_config.style_config,
        vocabularies,
        snap_to_bars=run_config.sections is None,
    )
    if run_config.sections is None:
        timeline, messages = validation
    else:
        messages = validation
    return timeline, tuple(messages)


def _meter_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for signature in style_config.allowed_meters:
        if signature in vocabularies.meters.label_map:
            ids.append(vocabularies.meters.token_for_label(signature).id)
    if not ids:
        ids.append(vocabularies.meters.token_for_id(0).id)
    return tuple(dict.fromkeys(ids))


def _key_anchor_ids(
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
    edo: Optional[int] = None,
) -> Tuple[int, ...]:
    resolved_edo = run_config.edo if edo is None else edo
    fifth = get_fifth_steps(resolved_edo) % resolved_edo
    anchors = (0, fifth, resolved_edo // 2)
    return tuple(dict.fromkeys(anchor % resolved_edo for anchor in anchors))


def _chord_id_for(key_id: int, quality: str, vocabularies: Vocabularies) -> int:
    for chord in vocabularies.chords:
        if chord.root_pc == key_id and chord.quality == quality:
            return chord.id
    return vocabularies.chords.token_for_id(0).id


def _groove_anchor_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for groove in vocabularies.grooves:
        if groove.family in style_config.groove_families:
            ids.append(groove.id)
    return tuple(dict.fromkeys(ids[: max(1, min(4, len(ids)))]))


def _is_legal_boundary_level(
    level: int,
    beat_in_bar: int,
    strong_beats: Sequence[int],
) -> bool:
    """Check whether a boundary level is legal at the given beat position.

    In GTTM / candidates:
      - Level 0 (none): legal anywhere.
      - Level 1 (local): requires a strong beat (beat_in_bar in strong_beats).
      - Level 2 (phrase) and Level 3 (section): require a downbeat (beat_in_bar == 0).
    """
    if level <= 0:
        return True
    if level == 1:
        return beat_in_bar in strong_beats
    return beat_in_bar == 0


def _highest_legal_boundary_level(
    beat_in_bar: int,
    strong_beats: Sequence[int],
    max_level: int = 3,
) -> int:
    """Return the highest legal boundary level for the given beat position."""
    for lvl in range(max_level, -1, -1):
        if _is_legal_boundary_level(lvl, beat_in_bar, strong_beats):
            return lvl
    return 0


def _endpoint_boundary_level(*, is_start: bool, beat_in_bar: int, strong_beats: Tuple[int, ...]) -> int:
    if beat_in_bar not in strong_beats:
        return 0
    if is_start:
        return 3
    return 2 if beat_in_bar == 0 else 1


def _candidate_score(
    state: BeatState,
    *,
    is_start: bool,
    boundary_level: int,
    primary_key_id: int,
    section: Optional[PlanningSection] = None,
    vocabularies: Optional[Vocabularies] = None,
    weights: Optional[PriorWeights] = None,
    target_tension: Optional[float] = None,
    edo: int = 12,
) -> float:
    """Score an endpoint candidate state against priors, boundary levels, and section targets.

    Mapping from SectionStyleWeights (scaled by lambda_section_style):
      - Meter: + sw.meter * lambda_section_style when state.meter is in section.allowed_meters.
      - Key region: + key_affinity * sw.key_region * lambda_section_style, where key_affinity
        is 1.0 / (1.0 + d_min) with d_min = min(tonal_distance(state_root, r, edo) for r in key_region)
        (or 1.0 if state.key_id == section.target_key_id).
      - Groove family: + sw.groove_family * lambda_section_style when state.groove matches section.groove_family.
      - Role: + sw.role * lambda_section_style when state.role is in section.preferred_roles.

    Target tension deviation (scaled by lambda_target_tension):
      - Gaussian energy: (tau_hat - tau_star)^2 / (2 * sigma_tension^2)
        evaluated via transition_target_tension_energy(rel_t, target_tension, weights.sigma_tension),
        subtracted as: - lambda_target_tension * e_tau.
    """
    score = 0.0
    score += 2.0 if state.beat_in_bar == 0 else 0.4
    score += 1.5 if state.boundary_lvl == boundary_level else 0.0
    score += 1.2 if state.key_id == primary_key_id else 0.5
    if is_start:
        score += 1.1 if state.role_id == 0 else 0.0
        score += 0.8 if state.head_id == 1 else 0.2
    else:
        score += 1.1 if state.role_id == 3 else 0.4
        score += 0.8 if state.head_id == 1 else 0.3

    if weights is not None and section is not None and vocabularies is not None:
        if weights.lambda_section_style > 0.0:
            sw = weights.style_weights
            if section.allowed_meters and vocabularies.meters.has_id(state.meter_id):
                meter_label = vocabularies.meters.token_for_id(state.meter_id).label
                if meter_label in section.allowed_meters:
                    score += sw.meter * weights.lambda_section_style
            if section.key_region and vocabularies.keys.has_id(state.key_id):
                state_root = vocabularies.keys.token_for_id(state.key_id).root_pc
                from aimusic.theory.tonal import tonal_distance
                d_min = min(tonal_distance(state_root, r, edo) for r in section.key_region)
                key_affinity = 1.0 / (1.0 + float(d_min))
                score += key_affinity * sw.key_region * weights.lambda_section_style
            elif section.target_key_id is not None and state.key_id == section.target_key_id:
                score += sw.key_region * weights.lambda_section_style
            if section.groove_family and vocabularies.grooves.has_id(state.groove_id):
                groove_tok = vocabularies.grooves.token_for_id(state.groove_id)
                if groove_tok.family == section.groove_family:
                    score += sw.groove_family * weights.lambda_section_style
            if section.preferred_roles and vocabularies.roles.has_id(state.role_id):
                role_label = vocabularies.roles.token_for_id(state.role_id).label
                if role_label in section.preferred_roles:
                    score += sw.role * weights.lambda_section_style
            if section.boundary_level > 0 and state.boundary_lvl != section.boundary_level:
                score -= sw.boundary * weights.lambda_section_style * abs(state.boundary_lvl - section.boundary_level)
            if state.beat_in_bar != 0:
                is_multi = vocabularies.meters is not None and len(vocabularies.meters) > 1
                align_factor = weights.multi_meter_alignment_factor if is_multi else 1.0
                score -= sw.start_alignment * weights.lambda_section_style * align_factor

        if weights.lambda_target_tension > 0.0 and target_tension is not None:
            from aimusic.scoring.tension import beat_tension, transition_target_tension_energy
            rel_t = beat_tension(None, state, vocabularies, edo)
            e_tau = transition_target_tension_energy(rel_t, target_tension, weights.sigma_tension)
            score -= weights.lambda_target_tension * e_tau

    return score


def _build_endpoint_distribution(
    *,
    time_index: int,
    beat_in_bar_by_meter: dict[int, int],
    is_start: bool,
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
    sections: Optional[Sequence[PlanningSection]] = None,
    timeline: Optional[SectionTimeline] = None,
    edo: Optional[int] = None,
) -> EndpointDistribution:
    resolved_edo = run_config.edo if edo is None else edo
    plan_config = run_config.plan_config

    if timeline is not None:
        resolved_timeline = timeline
    elif sections is not None:
        resolved_timeline = SectionTimeline(
            tuple(sections),
            total_beats=run_config.total_beats,
            edo=resolved_edo,
            multi_meter=len(run_config.style_config.allowed_meters) > 1,
        )
    else:
        sec_plan = build_section_plan(run_config)
        resolved_timeline = SectionTimeline(
            tuple(sec_plan),
            total_beats=run_config.total_beats,
            edo=resolved_edo,
            multi_meter=len(run_config.style_config.allowed_meters) > 1,
        )

    if is_start:
        sec = resolved_timeline.at_state(0)
        target_tension = float(resolved_timeline.target_tension(0))
    else:
        sec = resolved_timeline.at_state(run_config.total_beats)
        target_tension = float(resolved_timeline.target_tension(run_config.total_beats))

    has_style_guidance = (
        sec is not None
        and run_config.prior_weights is not None
        and run_config.prior_weights.lambda_section_style > 0.0
    )
    sw = run_config.prior_weights.style_weights if run_config.prior_weights else SectionStyleWeights()

    groove_ids = list(_groove_anchor_ids(run_config.style_config, vocabularies))
    if has_style_guidance and sec is not None and sec.groove_family and sw.groove_family > 0.0:
        sec_grooves = [g.id for g in vocabularies.grooves if g.family == sec.groove_family]
        if sec_grooves:
            groove_ids = list(dict.fromkeys(sec_grooves + groove_ids))

    key_ids = list(_key_anchor_ids(run_config, vocabularies, edo=resolved_edo))
    if has_style_guidance and sec is not None and sw.key_region > 0.0:
        if sec.key_region:
            sec_keys = [k for k in sec.key_region if vocabularies.keys.has_id(k)]
            if sec_keys:
                key_ids = list(dict.fromkeys(sec_keys + key_ids))
        elif sec.target_key_id is not None and vocabularies.keys.has_id(sec.target_key_id):
            key_ids = list(dict.fromkeys([sec.target_key_id] + key_ids))

    head_ids = (1, 2)
    chord_qualities = ("maj", "min")

    scored_candidates: list[tuple[float, BeatState]] = []
    allowed_meter_ids = _meter_ids(run_config.style_config, vocabularies)
    if has_style_guidance and sec is not None and sec.allowed_meters and sw.meter > 0.0:
        sec_meters = [
            vocabularies.meters.token_for_label(m).id
            for m in sec.allowed_meters
            if m in vocabularies.meters.label_map
        ]
        if sec_meters:
            allowed_meter_ids = tuple(dict.fromkeys(sec_meters + list(allowed_meter_ids)))

    for meter_id in allowed_meter_ids:
        beat_in_bar = beat_in_bar_by_meter[meter_id]
        strong_beats = vocabularies.meters.token_for_id(meter_id).strong_beats

        if is_start:
            base_lvl = _endpoint_boundary_level(is_start=True, beat_in_bar=beat_in_bar, strong_beats=strong_beats)
            if has_style_guidance and sec is not None and sw.boundary > 0.0:
                desired_level = sec.boundary_level
                if _is_legal_boundary_level(desired_level, beat_in_bar, strong_beats):
                    boundary_level = desired_level
                else:
                    boundary_level = _highest_legal_boundary_level(beat_in_bar, strong_beats)
                    sec_name = sec.name if sec is not None else "start"
                    _logger.info(
                        "Start endpoint boundary_level %d for section '%s' is illegal at beat_in_bar=%d; "
                        "falling back to highest legal level %d",
                        desired_level,
                        sec_name,
                        beat_in_bar,
                        boundary_level,
                    )
            else:
                boundary_level = base_lvl
        else:
            base_lvl = _endpoint_boundary_level(is_start=False, beat_in_bar=beat_in_bar, strong_beats=strong_beats)
            if has_style_guidance and sec is not None and sw.boundary > 0.0:
                desired_level = sec.boundary_level
                if _is_legal_boundary_level(desired_level, beat_in_bar, strong_beats):
                    boundary_level = desired_level
                else:
                    boundary_level = _highest_legal_boundary_level(beat_in_bar, strong_beats)
                    sec_name = sec.name if sec is not None else "end"
                    _logger.info(
                        "End endpoint boundary_level %d for section '%s' is illegal at beat_in_bar=%d; "
                        "falling back to highest legal level %d",
                        desired_level,
                        sec_name,
                        beat_in_bar,
                        boundary_level,
                    )
            else:
                boundary_level = base_lvl

        role_ids: tuple[int, ...]
        if is_start or boundary_level > 0:
            role_ids = (0, 1) if is_start else (3, 2)
        else:
            role_ids = (0, 1)

        if has_style_guidance and sec is not None and sec.preferred_roles and sw.role > 0.0:
            pref_roles = [
                vocabularies.roles.token_for_label(r).id
                for r in sec.preferred_roles
                if r in vocabularies.roles.label_map
                and not (not is_start and boundary_level >= 2 and r == "hold")
            ]
            if pref_roles:
                role_ids = tuple(dict.fromkeys(pref_roles + list(role_ids)))
        if not is_start and boundary_level >= 2 and hasattr(vocabularies, "roles") and vocabularies.roles:
            hold_token = vocabularies.roles.label_map.get("hold")
            if hold_token is not None:
                role_ids = tuple(r for r in role_ids if r != hold_token.id)


        for key_id in key_ids:
            for quality in chord_qualities:
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in role_ids:
                    for head_id in head_ids:
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            score = _candidate_score(
                                state,
                                is_start=is_start,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                                section=sec,
                                vocabularies=vocabularies,
                                weights=run_config.prior_weights,
                                target_tension=target_tension,
                                edo=resolved_edo,
                            )
                            score += (
                                run_config.plan_config.start_anchor_weight
                                if is_start
                                else run_config.plan_config.end_anchor_weight
                            )
                            scored_candidates.append((score, state))

    scored_candidates.sort(key=lambda item: (-item[0], _state_sort_key(item[1])))
    unique_states: list[BeatState] = []
    unique_scores: list[float] = []
    seen = set()
    for score, state in scored_candidates:
        if state in seen:
            continue
        seen.add(state)
        unique_states.append(state)
        unique_scores.append(score)
        if len(unique_states) >= plan_config.endpoint_top_k:
            break

    layer = Layer(time_index=time_index, states=tuple(unique_states))
    return EndpointDistribution(
        layer=layer,
        probabilities=_softmax(unique_scores, plan_config.endpoint_temperature),
    )



def generate_start_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    sections: Optional[Sequence[PlanningSection]] = None,
    timeline: Optional[SectionTimeline] = None,
    edo: Optional[int] = None,
) -> EndpointDistribution:
    resolved_edo = run_config.edo if edo is None else edo
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, resolved_edo
    )
    beat_positions = {meter_id: 0 for meter_id in _meter_ids(run_config.style_config, resolved_vocabs)}
    return _build_endpoint_distribution(
        time_index=0,
        beat_in_bar_by_meter=beat_positions,
        is_start=True,
        run_config=run_config,
        vocabularies=resolved_vocabs,
        sections=sections,
        timeline=timeline,
        edo=resolved_edo,
    )


def generate_end_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    sections: Optional[Sequence[PlanningSection]] = None,
    timeline: Optional[SectionTimeline] = None,
    edo: Optional[int] = None,
) -> EndpointDistribution:
    resolved_edo = run_config.edo if edo is None else edo
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, resolved_edo
    )
    beat_positions = {}
    for meter_id in _meter_ids(run_config.style_config, resolved_vocabs):
        beats_per_bar = resolved_vocabs.meters.token_for_id(meter_id).beats_per_bar
        beat_positions[meter_id] = run_config.total_beats % beats_per_bar
    return _build_endpoint_distribution(
        time_index=run_config.total_beats,
        beat_in_bar_by_meter=beat_positions,
        is_start=False,
        run_config=run_config,
        vocabularies=resolved_vocabs,
        sections=sections,
        timeline=timeline,
        edo=resolved_edo,
    )


def generate_method_a_endpoints(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    key: RNGKey,
    sample_endpoints: bool = False,
    sections: Optional[Sequence[PlanningSection]] = None,
    timeline: Optional[SectionTimeline] = None,
    edo: Optional[int] = None,
) -> tuple[MethodAEndpoints, RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    resolved_edo = run_config.edo if edo is None else edo
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, resolved_edo
    )
    resolved_sections = (
        tuple(sections)
        if sections is not None
        else (run_config.sections if run_config.sections is not None else build_section_plan(run_config))
    )
    resolved_timeline = (
        timeline
        if timeline is not None
        else SectionTimeline(
            resolved_sections,
            total_beats=run_config.total_beats,
            edo=resolved_edo,
            multi_meter=len(run_config.style_config.allowed_meters) > 1,
        )
    )
    pi0 = generate_start_endpoint_distribution(
        run_config,
        vocabularies=resolved_vocabs,
        sections=resolved_sections,
        timeline=resolved_timeline,
        edo=resolved_edo,
    )
    piT = generate_end_endpoint_distribution(
        run_config,
        vocabularies=resolved_vocabs,
        sections=resolved_sections,
        timeline=resolved_timeline,
        edo=resolved_edo,
    )
    start_choice, next_key = _choose_endpoint_state(
        pi0,
        key=key,
        sample=sample_endpoints,
    )
    end_choice, next_key = _choose_endpoint_state(
        piT,
        key=next_key,
        sample=sample_endpoints,
    )
    return MethodAEndpoints(
        pi0=pi0,
        piT=piT,
        start_choice=start_choice,
        end_choice=end_choice,
        sections=resolved_sections,
    ), next_key


def run_method_a(
    run_config: MethodARunConfig,
    *,
    key: RNGKey,
    prior: Optional[Prior] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[MethodAPlanResult, RNGKey]:
    """Run Method A from endpoint planning through SB path extraction."""
    _logger.info(f"Method A: {run_config.total_beats} beats, seed={run_config.seed}")
    tonal_context = build_tonal_context(
        run_config.edo,
        run_config.style_config,
        vocabularies=vocabularies,
    )
    resolved_vocabs = tonal_context.vocabularies
    resolved_sb = _resolved_sb_config(run_config)
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    stream_ids = (
        "endpoint_choice", "candidate_proposal", "bridge_sampling",
        "decoder.comping", "decoder.bass", "decoder.lead", "decoder.drums",
    )
    streams, next_key = allocate_named_keys(key, stream_ids)
    timeline, validation_messages = build_executable_section_timeline(
        run_config,
        resolved_vocabs,
    )
    resolved_sections = timeline.sections
    for message in validation_messages:
        _logger.warning(message)
    endpoints, _ = generate_method_a_endpoints(
        run_config,
        vocabularies=resolved_vocabs,
        key=streams["endpoint_choice"],
        sample_endpoints=run_config.use_sampling,
        sections=resolved_sections,
        timeline=timeline,
        edo=run_config.edo,
    )
    _logger.info(f"Endpoints: start={endpoints.start_choice.state} end={endpoints.end_choice.state}")
    start_endpoint = _singleton_endpoint_distribution(
        endpoints.start_choice.state,
        time_index=0,
    )
    end_endpoint = _singleton_endpoint_distribution(
        endpoints.end_choice.state,
        time_index=run_config.total_beats,
    )
    graph, _ = build_sparse_graph(
        start_layer=start_endpoint.layer,
        end_layer=end_endpoint.layer,
        total_beats=run_config.total_beats,
        sb_config=resolved_sb,
        style_config=run_config.style_config,
        vocabularies=resolved_vocabs,
        prior=NullPrior() if prior is None else prior,
        weights=run_config.prior_weights,
        edo=run_config.edo,
        key=streams["candidate_proposal"],
        d_max=resolved_sb.d_max,
        sections=endpoints.sections,
        timeline=timeline,
        section_guided_proposals=(
            run_config.plan_config.section_guided_proposals
            or resolved_sb.section_guided_proposals
        ),
    )
    _logger.info(f"Graph built: {len(graph.layers)} layers, {sum(len(l.states) for l in graph.layers)} states")
    aligned_endpoints = MethodAEndpoints(
        pi0=_align_endpoint_distribution(endpoints.pi0, graph.layers[0]),
        piT=_align_endpoint_distribution(endpoints.piT, graph.layers[-1]),
        start_choice=endpoints.start_choice,
        end_choice=endpoints.end_choice,
        sections=endpoints.sections,
    )
    problem = build_sb_problem(graph, start_endpoint, end_endpoint, sb_config=resolved_sb)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()
    _logger.info(f"SB solved: converged={solution.trace.converged}, iterations={solution.trace.iterations}")

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(bridge, streams["bridge_sampling"], include_edges=True, include_debug=True)
        path = sampled_path.path
        path_score = None
        _logger.info(f"Sampled path: {len(path) - 1} beats")
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None
        _logger.info(f"MAP path: {len(path) - 1} beats")

    from aimusic.scoring.gttm_features import FEATURE_REGISTRY

    trans_diags: list[TransitionDiagnostic] = []
    for t in range(len(path) - 1):
        source = path[t]
        target = path[t + 1]

        edge = None
        if t < len(graph.edges_by_time):
            for e in graph.edges_by_time[t]:
                if e.source == source and e.target == target:
                    edge = e
                    break
        if edge is None:
            raise ValueError(
                f"No matching edge in graph at time {t} for path transition {source} -> {target}"
            )

        edge_diag = None
        if graph.edge_diagnostics_by_time and t < len(graph.edge_diagnostics_by_time):
            for item in graph.edge_diagnostics_by_time[t]:
                if item.source == source and item.target == target:
                    edge_diag = item
                    break
        if edge_diag is None:
            raise ValueError(
                f"No matching edge score breakdown in graph at time {t} for path transition {source} -> {target}"
            )

        arrival_time = t + 1
        sec = timeline.at_state(arrival_time)
        sec_idx = timeline.section_index(arrival_time)
        is_start = timeline.is_section_start(arrival_time)

        family_breakdown: dict[str, float] = {
            "meter": 0.0,
            "grouping": 0.0,
            "harmonic": 0.0,
            "prolongational_role": 0.0,
            "melodic_head": 0.0,
            "groove": 0.0,
        }
        for fname, val in edge_diag.raw_feature_contributions.items():
            if fname in FEATURE_REGISTRY:
                family_breakdown[FEATURE_REGISTRY[fname].family] += float(val)

        target_tension_val = (
            float(edge_diag.target_tension)
            if edge_diag.target_tension is not None
            else float(timeline.target_tension(arrival_time))
        )
        realized_tension_val = (
            float(edge_diag.realized_tension)
            if edge_diag.realized_tension is not None
            else 0.0
        )

        diag = TransitionDiagnostic(
            time_index=t,
            arrival_time=arrival_time,
            section_name=sec.name,
            section_index=sec_idx,
            is_section_start=is_start,
            source_state=edge.source,
            target_state=edge.target,
            data_logp=float(edge_diag.data_logp),
            data_contribution=float(edge_diag.data_contribution),
            gttm_raw_features=dict(edge_diag.raw_feature_contributions),
            gttm_weighted_features=dict(edge_diag.weighted_feature_contributions),
            gttm_family_breakdown=family_breakdown,
            gttm_contribution=float(edge_diag.gttm_contribution),
            target_tension=target_tension_val,
            realized_tension=realized_tension_val,
            tension_deviation=float(edge_diag.tension_deviation),
            target_tension_energy=float(edge_diag.target_tension_energy),
            target_tension_contribution=float(edge_diag.target_tension_contribution),
            section_style_components=dict(edge_diag.section_style_components),
            section_style_energy=float(edge_diag.section_style_energy),
            section_style_contribution=float(edge_diag.section_style_contribution),
            final_log_weight=float(edge.log_weight),
            right_context_states=tuple(edge_diag.right_contexts),
        )
        try:
            diag.verify_decomposition(tol=1e-9)
        except Exception as exc:
            raise ValueError(
                f"Transition decomposition verification failed at time {t} (arrival {arrival_time}): {exc}"
            ) from exc
        trans_diags.append(diag)


    timeline = SectionTimeline(
        sections=endpoints.sections,
        total_beats=run_config.total_beats,
        edo=run_config.edo,
        multi_meter=len(run_config.style_config.allowed_meters) > 1,
    )

    diagnostics = MethodAPlanDiagnostics(
        section_tags=tuple(section.name for section in endpoints.sections),
        target_tension_arcs=tuple(section.target_tension_arc for section in endpoints.sections),
        chosen_start_state=endpoints.start_choice.state,
        chosen_end_state=endpoints.end_choice.state,
        endpoint_selection_mode=endpoints.start_choice.selection_mode,
        chosen_start_probability=endpoints.start_choice.selected_probability,
        chosen_end_probability=endpoints.end_choice.selected_probability,
        path_mode="sample" if run_config.use_sampling else "map",
        graph_layer_sizes=graph.diagnostics.layer_sizes,
        bridge_iterations=solution.trace.iterations,
        bridge_converged=solution.trace.converged,
        rng_stream_ids=stream_ids,
        transition_diagnostics=tuple(trans_diags),
        timeline=timeline,
    )
    return MethodAPlanResult(
        run_config=run_config,
        tonal_context=tonal_context,
        vocabularies=resolved_vocabs,
        endpoints=aligned_endpoints,
        graph=graph,
        sb_problem=problem,
        sb_solution=solution,
        bridge=bridge,
        path=path,
        path_score=path_score,
        sampled_path=sampled_path,
        diagnostics=diagnostics,
        path_edge_diagnostics=graph.diagnostics_for_path(path),
        timeline=timeline,
    ), next_key


def render_exact_bridge_demo(
    *,
    start_chord: str,
    end_chord: str,
    output_path: str,
    total_beats: int,
    seed: int = 0,
    meter: str = "4/4",
    groove: str = "straight_8ths",
    style_config: Optional[StyleConfig] = None,
    decode_config: Optional[DecodeConfig] = None,
    tempo_bpm: float = 120.0,
    edo: int = 12,
) -> ExactBridgeDemoResult:
    """Render a short bridge example using the current Method A pipeline.

    `start_chord`, `end_chord`, `meter`, and `groove` are accepted for compatibility with
    legacy scripts. The current implementation delegates endpoint selection to Method A.
    """
    del start_chord, end_chord, meter, groove

    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    run_config = MethodARunConfig(
        total_beats=total_beats,
        seed=seed,
        style_config=resolved_style,
        decode_config=resolved_decode,
        edo=edo,
    )
    plan_result, next_key = run_method_a(run_config, key=RNGKey(seed=seed))
    score, _ = decode_path_to_score(
        plan_result.path,
        decode_config=resolved_decode,
        vocabularies=plan_result.vocabularies,
        edo=plan_result.tonal_context.n,
        tempo_bpm=tempo_bpm,
        key=next_key,
    )
    from aimusic.render import render_midi
    from aimusic.theory.tonal import EDO

    render_midi(
        score,
        EDO(EDOConfig(n=plan_result.tonal_context.n, base_tuning=0)),
        output_path,
    )
    return ExactBridgeDemoResult(
        plan_result=plan_result,
        score=score,
        output_path=output_path,
    )
