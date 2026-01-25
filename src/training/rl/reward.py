"""
Reward Model for RLHF/PPO.

Calculates rewards for generated code based on correctness and quality metrics.
"""

import math


class CodeRewardModel:
    """
    Computes rewards for code generation tasks.
    """

    def __init__(self, complexity_penalty_weight: float = 0.05):
        self.complexity_penalty_weight = complexity_penalty_weight

    def compute_reward(self, code: str, test_cases: list[str], passed_tests: int) -> float:
        """
        Compute scalar reward.

        Args:
            code: Generated code string.
            test_cases: List of test case strings.
            passed_tests: Number of test cases passed.

        Returns:
            Scalar reward (e.g. 0.0 to 1.0).
        """
        if not test_cases:
            return 0.0

        # 1. Correctness Reward (Pass Rate)
        pass_rate = passed_tests / len(test_cases)
        correctness_reward = pass_rate * 1.0  # Scale to [0, 1]

        # 2. Complexity Penalty (Simple Length Heuristic for now)
        # Prefer concise code.
        num_lines = len(code.split("\n"))
        complexity_penalty = math.log(max(1, num_lines)) * self.complexity_penalty_weight

        # Total Reward
        # If pass_rate is 0, penalty should not make it negative?
        # Actually RL handles negative rewards fine.
        total_reward = correctness_reward - complexity_penalty

        return total_reward
