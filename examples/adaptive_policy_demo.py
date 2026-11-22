"""Run a small, deterministic demonstration of the adaptive-policy interface."""

from agents.adaptive_challenge_policy import (
    AdaptiveChallengePolicy,
    LearnerState,
)


def main() -> None:
    # Illustrative values only; replace them with validated learner assessments.
    learner = LearnerState(
        analysis=0.55,
        evidence_evaluation=0.40,
        inference=0.62,
        counterargument=0.35,
        self_correction=0.50,
        open_mindedness=0.60,
        claim_evidence_coverage=0.45,
        support_strength=0.50,
        rebuttal_quality=0.30,
        contradiction_rate=0.25,
        previous_difficulty=0.50,
        learning_progress=0.20,
    )

    # This is a transparent baseline, not a trained RL agent.
    policy = AdaptiveChallengePolicy.rule_based()
    decision = policy.select_challenge(learner)

    print(f"Policy:     {decision.policy_source}")
    print(f"Persona:    {decision.persona}")
    print(f"Difficulty: {decision.difficulty}")
    print(f"Focus:      {decision.focus}")
    print("\nInstruction for the existing Debate Agent:")
    print(decision.instruction)


if __name__ == "__main__":
    main()
