from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from concordia.errors import ValidationError


@dataclass(frozen=True)
class PPOTrainingResult:
    epochs: int
    final_mean_reward: float
    final_entropy: float


class PPOEligibilityPolicy:
    """Compact masked PPO policy for recommendation-eligibility actions only."""

    def __init__(self, state_dimension: int, action_count: int, seed: int = 0) -> None:
        if state_dimension < 1 or action_count < 2:
            raise ValidationError("PPO state/action dimensions are invalid")
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0.0, 0.01, (state_dimension, action_count))
        self.bias = np.zeros(action_count, dtype=float)
        self.value_weights = np.zeros(state_dimension, dtype=float)
        self.value_bias = 0.0

    def probabilities(self, states: np.ndarray, valid_actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        valid_actions = np.asarray(valid_actions, dtype=bool)
        if states.ndim != 2 or valid_actions.shape != (len(states), self.bias.size):
            raise ValidationError("PPO state/action-mask shapes are inconsistent")
        if np.any(~valid_actions.any(axis=1)):
            raise ValidationError("every PPO state requires at least one valid action")
        logits = states @ self.weights + self.bias
        logits = np.where(valid_actions, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        exponent = np.exp(logits) * valid_actions
        return exponent / exponent.sum(axis=1, keepdims=True)

    def fit(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        valid_actions: np.ndarray,
        *,
        epochs: int,
        update_epochs: int,
        learning_rate: float,
        value_learning_rate: float,
        clip_ratio: float,
        entropy_coefficient: float,
    ) -> PPOTrainingResult:
        states = np.asarray(states, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        valid_actions = np.asarray(valid_actions, dtype=bool)
        if rewards.shape != valid_actions.shape:
            raise ValidationError("PPO reward and action-mask shapes must match")
        last_reward = last_entropy = 0.0
        for _ in range(epochs):
            old_probabilities = self.probabilities(states, valid_actions)
            actions = np.asarray(
                [self.rng.choice(self.bias.size, p=row) for row in old_probabilities],
                dtype=int,
            )
            observed = rewards[np.arange(len(states)), actions]
            values = states @ self.value_weights + self.value_bias
            advantages = observed - values
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
            old_selected = old_probabilities[np.arange(len(states)), actions]
            one_hot = np.eye(self.bias.size)[actions]
            for _update in range(update_epochs):
                probabilities = self.probabilities(states, valid_actions)
                selected = probabilities[np.arange(len(states)), actions]
                ratio = selected / np.maximum(old_selected, 1e-12)
                active = np.where(
                    advantages >= 0,
                    ratio <= 1.0 + clip_ratio,
                    ratio >= 1.0 - clip_ratio,
                )
                coefficient = np.where(active, advantages * ratio, 0.0)
                score = one_hot - probabilities
                gradient = coefficient[:, None] * score
                entropy_gradient = -(
                    probabilities * (np.log(np.maximum(probabilities, 1e-12)) + 1.0)
                )
                self.weights += learning_rate * (
                    states.T @ (gradient + entropy_coefficient * entropy_gradient)
                ) / len(states)
                self.bias += learning_rate * (
                    gradient + entropy_coefficient * entropy_gradient
                ).mean(axis=0)
            value_error = observed - (states @ self.value_weights + self.value_bias)
            self.value_weights += value_learning_rate * (states.T @ value_error) / len(states)
            self.value_bias += value_learning_rate * float(value_error.mean())
            last_reward = float(observed.mean())
            last_entropy = float(
                -np.mean(np.sum(old_probabilities * np.log(old_probabilities + 1e-12), axis=1))
            )
        return PPOTrainingResult(epochs, last_reward, last_entropy)

    def act(self, states: np.ndarray, valid_actions: np.ndarray) -> np.ndarray:
        return np.argmax(self.probabilities(states, valid_actions), axis=1)
