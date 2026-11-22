# CritiqAI Adaptive Challenge Policy (RL Extension)

This add-on introduces an **optional learner-state-aware challenge-selection layer**
for CritiqAI. It is designed to sit beside the existing multi-agent system, not to
replace the Orchestrator, Debate Agent, Challenge Validator, scorer MCP server,
Analytics/Report Agent, or teacher approval workflow.

## What it adds

- A validated 12-feature learner-state interface.
- A discrete action space containing 4 existing challenger personas × 3 difficulty levels.
- A transparent rule-based baseline for smoke tests and baseline experiments.
- An inference wrapper for a **trained Stable-Baselines3 PPO checkpoint**.
- A potential function and a Potential-Based Reward Shaping (PBRS) helper.
- Unit tests and a small demonstration script.

**Important:** the PPO model is not included and is not pretrained by this add-on.
The rule-based policy is not reinforcement learning. To use PPO, train a policy
against a Gymnasium environment built from real, ethically collected learner
transitions and independently assessed learning outcomes.

## Where to put the files

Copy the contents of this add-on into the root of your existing CritiqAI repository.
The `agents/` directory already exists in that project.

```text
CritqAI/
├── agents/
│   ├── ... existing agent modules ...
│   ├── __init__.py
│   └── adaptive_challenge_policy.py
├── examples/
│   └── adaptive_policy_demo.py
├── tests/
│   └── test_adaptive_challenge_policy.py
├── requirements-rl.txt
└── README_ADAPTIVE_RL.md
```

If your repository already has `agents/__init__.py`, **keep your existing file**;
do not overwrite it with the minimal file in this add-on.

## Quick start: baseline mode

The baseline requires only the Python standard library.

```bash
python examples/adaptive_policy_demo.py
```

The script constructs an example learner profile and prints the selected persona,
difficulty, and instruction for the existing Debate Agent. This verifies the
interface only; it is not a learning experiment and does not demonstrate an
improvement in student outcomes.

## PPO inference

Install the optional dependencies:

```bash
pip install -r requirements-rl.txt
```

After training a compatible PPO model, load it as follows:

```python
from agents.adaptive_challenge_policy import (
    AdaptiveChallengePolicy,
    LearnerState,
)

state = LearnerState(
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

policy = AdaptiveChallengePolicy.from_ppo("models/critiqai_ppo")
decision = policy.select_challenge(state)

# Pass decision.instruction to the existing Debate Agent.
print(decision.persona, decision.difficulty)
print(decision.instruction)
```

The checkpoint must have been trained with exactly the same observation ordering
and action mapping documented below. A mismatched training environment makes
inference invalid even if the checkpoint loads successfully.

## State and action contract

### Observation: 12 values, each normalized to [0, 1]

| Index | Feature | Meaning |
|---:|---|---|
| 0 | `analysis` | Learner's analysis skill estimate |
| 1 | `evidence_evaluation` | Ability to assess evidence |
| 2 | `inference` | Quality of inference |
| 3 | `counterargument` | Ability to formulate/respond to counterarguments |
| 4 | `self_correction` | Willingness/ability to revise reasoning |
| 5 | `open_mindedness` | Consideration of alternative views |
| 6 | `claim_evidence_coverage` | Argument-graph claim/evidence coverage |
| 7 | `support_strength` | Strength of support relations |
| 8 | `rebuttal_quality` | Quality of rebuttal relations |
| 9 | `contradiction_rate` | Detected contradiction rate (higher is worse) |
| 10 | `previous_difficulty` | Previous difficulty: 0, 0.5, or 1 |
| 11 | `learning_progress` | Normalized progress estimate |

These are **input fields**, not measurements automatically produced by the
existing notebook. The notebook's Paul-Elder scorer currently exposes four
dimensions (`logical_coherence`, `evidence_quality`,
`counterargument_handling`, `scope_awareness`). Those four values are not
automatically equivalent to all six skill fields and four graph fields above.
A validated feature-extraction/mapping layer must be implemented before using
real learner records. Do not fill missing features with arbitrary values in a
research experiment.

### Action: 12 discrete choices

`action_id = persona_index * 3 + difficulty_index`

| Persona index | Persona | Difficulty index 0 | 1 | 2 |
|---:|---|---|---|---|
| 0 | Skeptic | low (0.0) | medium (0.5) | high (1.0) |
| 1 | Devil's Advocate | low (0.0) | medium (0.5) | high (1.0) |
| 2 | Nitpicker | low (0.0) | medium (0.5) | high (1.0) |
| 3 | Expander | low (0.0) | medium (0.5) | high (1.0) |

The selected instruction preserves CritiqAI's pedagogical constraint: challenge
the learner, do not provide a correction, solution, or model answer.

## Integration with the current CritiqAI pipeline

The add-on returns one persona for **one challenge decision**. It should be called
when a new learner-state estimate is available, for example between debate turns:

1. Keep the current Orchestrator and existing agents.
2. Obtain the learner's latest assessed state from a validated feature extractor.
3. Call `policy.select_challenge(state)`.
4. Pass `decision.instruction` (and, if useful, `decision.persona`) to the Debate Agent.
5. Keep the existing Challenge Validator in the loop.
6. Log the state, selected action, validator result, student response, and
   independently computed reward for later analysis.

### Important compatibility note

The supplied Kaggle notebook describes the PersonaSelector as selecting **two**
personas, whereas the README architecture text supplied with the project describes
selecting **one** persona. This extension does not silently change either behavior.
Use it as a per-turn adaptive selector alongside the current session-level
PersonaSelector, or explicitly decide and document a one-persona-per-turn design
before replacing the existing selector.

Do not pass raw learner essays or personal data to an external training pipeline
without appropriate consent, data minimization, and institutional approval.

## PBRS helper

The module implements:

\[
r'_t = r_t + \\eta(\\gamma\\Phi(s_{t+1}) - \\Phi(s_t))
\]

where `task_reward` must come from the real task assessment process. The helper
uses a bounded, transparent proxy potential based on the supplied skill and
argument-structure features. It is a **starting implementation**, not a validated
measure of learning gain. The shaping discount factor must match the RL algorithm's
discount factor. Terminal transitions use zero next-state potential.

For a research paper, measure learning gain with an independent post-test or
transfer assessment on an unseen topic. Do not treat an increase in the in-session
rubric score alone as proof of durable learning.

## Tests

From the repository root:

```bash
python -m unittest discover -s tests -p "test_adaptive_challenge_policy.py" -v
```

## Research and reproducibility checklist

Before making empirical claims:

- Define and validate how each state feature is computed.
- Collect learner transitions with consent and appropriate ethics review.
- Use learner-level train/validation/test splits (never split turns from the
  same learner across partitions).
- Define a task reward from a pre-registered rubric and independent assessment.
- Compare against the existing static PersonaSelector and a rule-based baseline.
- Include PPO without PBRS and PPO with PBRS as separate conditions.
- Report inter-rater reliability, transfer performance, retention if available,
  latency, token cost, and failure/validator-retry rates.
- Run ablations for state features and difficulty adaptation.
- Publish seeds, environment configuration, and evaluation protocol.

## Files

- `agents/adaptive_challenge_policy.py`: policy, state schema, action mapping, PBRS helper.
- `examples/adaptive_policy_demo.py`: baseline usage example.
- `tests/test_adaptive_challenge_policy.py`: deterministic unit tests.
- `requirements-rl.txt`: optional RL inference/training dependencies.
- `README_ADAPTIVE_RL.md`: this integration and research guide.
