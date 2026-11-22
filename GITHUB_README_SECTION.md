# Suggested README addition

Paste this section into the main CritiqAI README under a heading such as
`## Adaptive Challenge Selection (Experimental)`:

> CritiqAI includes an optional adaptive challenge-selection extension. It
> represents a learner's current reasoning profile as a normalized state vector
> and selects a challenger persona and difficulty for the next debate turn.
> The extension provides a transparent rule-based baseline and an inference
> interface for a separately trained PPO policy. It also includes a PBRS helper
> for RL experiments. The PPO checkpoint and validated learner-state extraction
> pipeline are not bundled; empirical learning-gain claims require a controlled
> evaluation with independent assessment.

Link the detailed guide with:
`[Adaptive RL extension and integration guide](README_ADAPTIVE_RL.md)`

# Suggested commit message

feat: add experimental adaptive challenge policy with PPO and PBRS support
