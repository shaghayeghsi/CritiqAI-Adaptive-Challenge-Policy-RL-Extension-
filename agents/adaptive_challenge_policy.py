"""
CritiqAI: Adaptive Socratic Challenge Policy
===========================================

A small integration module for selecting the next debate persona and
difficulty from a learner's reasoning profile.

Observation schema (12 normalized values):
  0-5: analysis, evidence evaluation, inference, counterargument,
       self-correction, open-mindedness
  6-9: claim/evidence coverage, support strength, rebuttal quality,
       contradiction rate
  10: previous difficulty; 11: learning progress

Action schema: 4 personas x 3 difficulty levels = 12 discrete actions.
Action id = persona_index * 3 + difficulty_index.

PPO inference is supported through Stable-Baselines3. The transparent
rule-based policy is provided only as a baseline/smoke-test option.
This module does not fabricate training data or claim that a PPO model
has been trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
import math


STATE_DIM = 12
ACTION_COUNT = 12

SKILL_NAMES = (
    "analysis", "evidence_evaluation", "inference",
    "counterargument", "self_correction", "open_mindedness",
)


class Persona(IntEnum):
    SKEPTIC = 0
    DEVILS_ADVOCATE = 1
    NITPICKER = 2
    EXPANDER = 3


PERSONA_NAMES = {
    Persona.SKEPTIC: "Skeptic",
    Persona.DEVILS_ADVOCATE: "Devil's Advocate",
    Persona.NITPICKER: "Nitpicker",
    Persona.EXPANDER: "Expander",
}

PERSONA_FOCUS = {
    Persona.SKEPTIC: "test the sufficiency and reliability of the evidence",
    Persona.DEVILS_ADVOCATE: "examine a plausible opposing position",
    Persona.NITPICKER: "identify ambiguity, unsupported inference, or a logical gap",
    Persona.EXPANDER: "explore implications, assumptions, and alternative explanations",
}


@dataclass(frozen=True)
class LearnerState:
    """Validated, normalized learner state used by the policy."""

    analysis: float
    evidence_evaluation: float
    inference: float
    counterargument: float
    self_correction: float
    open_mindedness: float
    claim_evidence_coverage: float
    support_strength: float
    rebuttal_quality: float
    contradiction_rate: float
    previous_difficulty: float
    learning_progress: float

    def __post_init__(self) -> None:
        for name, value in self.as_mapping().items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1].")

    def as_mapping(self) -> dict[str, float]:
        return {
            "analysis": self.analysis,
            "evidence_evaluation": self.evidence_evaluation,
            "inference": self.inference,
            "counterargument": self.counterargument,
            "self_correction": self.self_correction,
            "open_mindedness": self.open_mindedness,
            "claim_evidence_coverage": self.claim_evidence_coverage,
            "support_strength": self.support_strength,
            "rebuttal_quality": self.rebuttal_quality,
            "contradiction_rate": self.contradiction_rate,
            "previous_difficulty": self.previous_difficulty,
            "learning_progress": self.learning_progress,
        }

    def to_observation(self) -> list[float]:
        values = list(self.as_mapping().values())
        if len(values) != STATE_DIM:
            raise RuntimeError("Learner-state schema does not match STATE_DIM.")
        return values


@dataclass(frozen=True)
class ChallengeDecision:
    """Auditable decision to pass to CritiqAI's existing Debate Agent."""

    action_id: int
    persona: str
    difficulty: str
    difficulty_value: float
    focus: str
    instruction: str
    policy_source: str


class PolicyModel(Protocol):
    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        ...


def decode_action(action_id: int) -> tuple[Persona, str, float]:
    """Decode one of the 12 discrete policy actions."""
    if not 0 <= int(action_id) < ACTION_COUNT:
        raise ValueError(f"action_id must be in [0, {ACTION_COUNT - 1}].")
    persona_index, difficulty_index = divmod(int(action_id), 3)
    label, value = (("low", 0.0), ("medium", 0.5), ("high", 1.0))[difficulty_index]
    return Persona(persona_index), label, value


def encode_action(persona: Persona, difficulty: str) -> int:
    levels = {"low": 0, "medium": 1, "high": 2}
    try:
        return int(persona) * 3 + levels[difficulty.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError("Unknown persona or difficulty.") from exc


def _instruction(persona: Persona, difficulty: str) -> str:
    intensity = {
        "low": "Ask one accessible question about a clear reasoning gap.",
        "medium": "Ask one probing question requiring justification and analysis.",
        "high": (
            "Ask one demanding question that tests the argument against a "
            "plausible counterexample or competing interpretation."
        ),
    }[difficulty]
    return (
        f"Act as the {PERSONA_NAMES[persona]}. Your focus is to "
        f"{PERSONA_FOCUS[persona]}. {intensity} Do not provide an answer, "
        "correction, model argument, or solution. Ask exactly one question."
    )


class RuleBasedBaseline:
    """Interpretable baseline; it is not a learned reinforcement-learning policy."""

    PERSONA_FOR_SKILL = {
        "analysis": Persona.NITPICKER,
        "evidence_evaluation": Persona.SKEPTIC,
        "inference": Persona.NITPICKER,
        "counterargument": Persona.DEVILS_ADVOCATE,
        "self_correction": Persona.SKEPTIC,
        "open_mindedness": Persona.EXPANDER,
    }

    def predict(self, observation: Sequence[float], deterministic: bool = True) -> int:
        if len(observation) != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM} observation values.")
        weakest = min(range(6), key=lambda i: observation[i])
        persona = self.PERSONA_FOR_SKILL[SKILL_NAMES[weakest]]
        mean_skill = sum(observation[:6]) / 6.0
        difficulty_index = 0 if mean_skill < 0.40 else 1 if mean_skill < 0.70 else 2
        return int(persona) * 3 + difficulty_index


class AdaptiveChallengePolicy:
    """Policy wrapper with explicit PPO and rule-based construction paths."""

    def __init__(
        self,
        model: PolicyModel,
        *,
        policy_source: str,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.policy_source = policy_source
        self.deterministic = deterministic

    @classmethod
    def from_ppo(
        cls, checkpoint_path: str | Path, *, deterministic: bool = True
    ) -> "AdaptiveChallengePolicy":
        """Load a trained Stable-Baselines3 PPO checkpoint (inference only)."""
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "Install PPO inference dependencies with: "
                "pip install stable-baselines3 gymnasium numpy"
            ) from exc

        path = Path(checkpoint_path)
        if not path.exists() and not Path(f"{path}.zip").exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        model = PPO.load(str(path), device="cpu")
        return cls(model, policy_source=f"ppo:{path}", deterministic=deterministic)

    @classmethod
    def rule_based(cls) -> "AdaptiveChallengePolicy":
        return cls(RuleBasedBaseline(), policy_source="rule_based_baseline")

    def select_challenge(self, state: LearnerState) -> ChallengeDecision:
        prediction = self.model.predict(
            state.to_observation(), deterministic=self.deterministic
        )
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        if hasattr(action, "item"):
            action = action.item()
        action_id = int(action)
        persona, difficulty, difficulty_value = decode_action(action_id)
        return ChallengeDecision(
            action_id=action_id,
            persona=PERSONA_NAMES[persona],
            difficulty=difficulty,
            difficulty_value=difficulty_value,
            focus=PERSONA_FOCUS[persona],
            instruction=_instruction(persona, difficulty),
            policy_source=self.policy_source,
        )


def learner_potential(state: LearnerState) -> float:
    """Bounded proxy potential; validate it against independent learning gains."""
    skill_score = sum(state.to_observation()[:6]) / 6.0
    graph_score = (
        state.claim_evidence_coverage
        + state.support_strength
        + state.rebuttal_quality
        + (1.0 - state.contradiction_rate)
    ) / 4.0
    return 0.70 * skill_score + 0.30 * graph_score


def potential_based_shaping_reward(
    task_reward: float,
    current_state: LearnerState,
    next_state: LearnerState,
    *,
    discount_factor: float,
    shaping_scale: float = 1.0,
    terminal: bool = False,
) -> float:
    """PBRS: r' = r + eta * (gamma * Phi(s') - Phi(s))."""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not math.isfinite(shaping_scale) or shaping_scale < 0.0:
        raise ValueError("shaping_scale must be finite and non-negative.")
    phi_now = learner_potential(current_state)
    phi_next = 0.0 if terminal else learner_potential(next_state)
    return float(
        task_reward
        + shaping_scale * (discount_factor * phi_next - phi_now)
    )


def state_from_mapping(values: Mapping[str, float]) -> LearnerState:
    """Create a state from scorer/analytics output with explicit field checks."""
    names = tuple(LearnerState.__dataclass_fields__)
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"Missing learner-state fields: {', '.join(missing)}")
    return LearnerState(**{name: float(values[name]) for name in names})


__all__ = [
    "AdaptiveChallengePolicy",
    "ChallengeDecision",
    "LearnerState",
    "Persona",
    "RuleBasedBaseline",
    "decode_action",
    "encode_action",
    "learner_potential",
    "potential_based_shaping_reward",
    "state_from_mapping",
]
