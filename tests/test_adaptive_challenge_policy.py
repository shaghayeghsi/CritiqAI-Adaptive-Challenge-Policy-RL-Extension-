import unittest

from agents.adaptive_challenge_policy import (
    AdaptiveChallengePolicy,
    LearnerState,
    Persona,
    decode_action,
    encode_action,
    learner_potential,
    potential_based_shaping_reward,
)


def sample_state(**overrides):
    values = dict(
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
    values.update(overrides)
    return LearnerState(**values)


class AdaptivePolicyTests(unittest.TestCase):
    def test_state_rejects_out_of_range_feature(self):
        with self.assertRaises(ValueError):
            sample_state(analysis=1.5)

    def test_all_actions_round_trip(self):
        for action in range(12):
            persona, difficulty, _ = decode_action(action)
            self.assertEqual(encode_action(persona, difficulty), action)

    def test_baseline_returns_valid_decision(self):
        decision = AdaptiveChallengePolicy.rule_based().select_challenge(sample_state())
        self.assertIn(decision.persona, {
            "Skeptic", "Devil's Advocate", "Nitpicker", "Expander"
        })
        self.assertIn(decision.difficulty, {"low", "medium", "high"})
        self.assertIn("Do not provide an answer", decision.instruction)

    def test_potential_is_bounded(self):
        self.assertGreaterEqual(learner_potential(sample_state()), 0.0)
        self.assertLessEqual(learner_potential(sample_state()), 1.0)

    def test_terminal_transition_uses_zero_next_potential(self):
        state = sample_state()
        gamma = 0.9
        eta = 0.5
        reward = 1.0
        shaped = potential_based_shaping_reward(
            reward, state, sample_state(analysis=0.9),
            discount_factor=gamma, shaping_scale=eta, terminal=True
        )
        expected = reward - eta * learner_potential(state)
        self.assertAlmostEqual(shaped, expected)

    def test_action_range_is_checked(self):
        with self.assertRaises(ValueError):
            decode_action(12)


if __name__ == "__main__":
    unittest.main()
