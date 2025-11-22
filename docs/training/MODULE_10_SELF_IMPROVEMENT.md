# Module 10: Self-Improving AI Systems

**Duration:** 16 hours (4 days)
**Format:** Advanced Workshop + Capstone Project
**Difficulty:** Expert
**Prerequisites:** Completed Modules 1-9, understanding of reinforcement learning and MCTS

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement AlphaZero-style self-play** training pipelines
2. **Design RLHF systems** for model alignment
3. **Build A/B testing infrastructure** for model evaluation
4. **Monitor deployed systems** with comprehensive metrics
5. **Create end-to-end self-improvement loops** for production AI

---

## Session 1: AlphaZero-Style Self-Play (4 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [training/self_play_generator.py](../../training/self_play_generator.py) - Complete self-play implementation
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [MuZero](https://arxiv.org/abs/1911.08265) - Extension to unknown dynamics

### Lecture: Self-Play Training Philosophy (90 minutes)

#### Why Self-Play?

**Key Insight:** The best training data comes from the model itself

**AlphaZero Breakthrough:**
- No human data required
- Learns from scratch through self-play
- Surpasses human expert level
- Discovers novel strategies

**Application to Multi-Agent MCTS:**
- HRM agent generates task decompositions
- TRM agent refines solutions
- MCTS explores solution space
- System learns from its own executions

#### AlphaZero Training Loop

**Core Cycle:**
```
Current Model (Policy π, Value V)
    ↓
1. Self-Play Episodes
   - Generate N games/tasks using current model
   - Record complete MCTS search trees
   - Store (state, π_mcts, z) tuples
    ↓
2. Training Data Extraction
   - Policy targets: π_mcts (visit count distribution)
   - Value targets: z (actual game outcome)
   - Multiple examples per episode
    ↓
3. Model Training
   - Train on policy + value targets
   - Loss = MSE(V, z) + CrossEntropy(π, π_mcts)
   - Multiple epochs on replay buffer
    ↓
4. Evaluation
   - New model vs old model
   - Win rate >55% → Accept new model
    ↓
5. Iteration
   - Replace old model with new model
   - Repeat from step 1
```

**Implementation Overview:**
```python
from training.self_play_generator import (
    SelfPlayTrainer,
    SelfPlayEpisodeGenerator,
    TrainingDataExtractor
)

# Initialize trainer
trainer = SelfPlayTrainer(
    hrm_agent=hrm_model,  # Your HRM model
    trm_agent=trm_model,  # Your TRM model
    config={
        "games_per_iteration": 1000,
        "batch_size": 64,
        "parallel_batch_size": 32,
        "max_buffer_size": 10000,
        "mcts": {
            "num_simulations": 100,
            "c_puct": 1.25,
        },
    },
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run training iterations
for iteration in range(num_iterations):
    metrics = await trainer.iteration(iteration)

    print(f"Iteration {iteration}:")
    print(f"  Episodes: {metrics['num_episodes']}")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Policy examples: {metrics['num_policy_examples']}")
    print(f"  Eval success rate: {metrics['eval_success_rate']:.2%}")
```

### Lecture: Episode Generation (60 minutes)

#### Task Generation

**Built-in Task Generators:**
1. **MathProblemGenerator:** Arithmetic, algebra, calculus
2. **CodeGenerationTaskGenerator:** Function, algorithm, class design
3. **MultiStepReasoningGenerator:** Logic, planning, constraint satisfaction
4. **MCTSSearchTaskGenerator:** Game states, optimization, path finding

**Example Task Generation:**
```python
from training.self_play_generator import (
    MathProblemGenerator,
    CodeGenerationTaskGenerator,
    MCTSSearchTaskGenerator
)

# Math problems
math_gen = MathProblemGenerator(
    difficulty_range=(0.1, 1.0),
    seed=42
)
math_tasks = math_gen.generate(num_tasks=100)

# Example task
task = math_tasks[0]
# {
#     "task_id": "math_0_45",
#     "type": "algebra",
#     "difficulty": 0.45,
#     "problem": "Solve for x: 3x + 5 = 17",
#     "answer": 4.0,
#     "steps": ["3x = 12", "x = 4.0"]
# }

# Code generation
code_gen = CodeGenerationTaskGenerator(
    difficulty_range=(0.3, 0.8)
)
code_tasks = code_gen.generate(num_tasks=50)

# MCTS search tasks
mcts_gen = MCTSSearchTaskGenerator(
    difficulty_range=(0.4, 1.0)
)
search_tasks = mcts_gen.generate(num_tasks=100)
```

#### Episode Structure

**Complete Episode Trace:**
```python
@dataclass
class SelfPlayEpisode:
    """Complete episode from self-play execution."""

    task_id: str
    initial_state: Any
    actions: list[Action]  # All actions taken
    states: list[State]    # State after each action
    rewards: list[float]   # Reward for each transition
    mcts_traces: list[MCTSTrace]  # MCTS search info at each step
    outcome: str  # "success", "failure", "timeout", "error"
    metadata: dict[str, Any]

    # Computed fields
    total_reward: float
    episode_length: int
    search_efficiency: float  # reward / (simulations * time)
    solution_path: list[str]  # state_ids on success path
```

**MCTS Trace Capture:**
```python
@dataclass
class MCTSTrace:
    """Captures MCTS search tree information at a decision point."""

    root_state_id: str
    num_simulations: int
    visit_counts: dict[str, int]      # action_id -> visit count
    q_values: dict[str, float]        # action_id -> Q-value
    prior_probs: dict[str, float]     # action_id -> prior probability
    selected_action: str
    tree_depth: int
    search_time: float
    value_estimates: dict[str, float] # state_id -> value
```

### Lecture: Training Data Extraction (60 minutes)

#### Policy Training Examples

**From MCTS Visit Counts:**
```python
# MCTS search at decision point
mcts_trace = MCTSTrace(
    root_state_id="state_5",
    num_simulations=100,
    visit_counts={
        "action_A": 60,  # Most visited
        "action_B": 30,
        "action_C": 10,
    },
    q_values={
        "action_A": 0.85,
        "action_B": 0.72,
        "action_C": 0.45,
    },
    selected_action="action_A"
)

# Convert to improved policy target
policy_target = {
    "action_A": 60/100,  # 0.60
    "action_B": 30/100,  # 0.30
    "action_C": 10/100,  # 0.10
}

# Training example: (state, policy_target)
example = TrainingExample(
    example_id="episode_123_policy_5",
    example_type="policy",
    state=state_tensor,
    target=policy_target,
    weight=1.0,  # Full weight for successful episodes
)
```

**Key Insight:** MCTS visit counts represent an improved policy that the neural network should learn to approximate directly.

#### Value Training Examples

**From Episode Outcomes:**
```python
# Episode with rewards
episode = SelfPlayEpisode(
    rewards=[0.1, 0.2, 0.3, 1.0],  # Final reward = success
    outcome="success"
)

# Compute discounted returns (gamma=0.99)
returns = []
G = 0.0
for reward in reversed(episode.rewards):
    G = reward + 0.99 * G
    returns.insert(0, G)

# returns = [1.53, 1.47, 1.27, 1.0]

# Training examples: (state, value_target)
for state, value_target in zip(episode.states, returns):
    example = TrainingExample(
        example_id=f"episode_{episode.task_id}_value_{i}",
        example_type="value",
        state=state.representation,
        target=value_target,  # Expected cumulative reward
        weight=1.0
    )
```

#### Negative Examples

**From Failed Episodes:**
```python
def extract_negative_examples(episode: SelfPlayEpisode) -> list[TrainingExample]:
    """Extract negative examples from failures."""
    if episode.outcome != "success":
        # Find worst decisions
        worst_states = np.argsort(episode.rewards)[:3]

        examples = []
        for idx in worst_states:
            trace = episode.mcts_traces[idx]

            # Negative example: avoid selected action
            negative_policy = {trace.selected_action: 0.0}

            examples.append(TrainingExample(
                example_type="negative",
                state=episode.states[idx].representation,
                target=negative_policy,
                weight=0.3,  # Lower weight for negative examples
                metadata={
                    "outcome": episode.outcome,
                    "reward": episode.rewards[idx]
                }
            ))

        return examples
```

### Hands-On Exercise: Generate Training Episodes (60 minutes)

**Exercise 10.1: Run Self-Play Episode Generation**

**Objective:** Generate 1,000 self-play episodes and extract training data.

**Requirements:**
1. Configure task generators (math, code, reasoning)
2. Run self-play episode generation
3. Extract policy and value training examples
4. Analyze episode statistics
5. Save training dataset

**Template:**
```python
# labs/module_10/exercise_10_1_self_play.py

import asyncio
from training.self_play_generator import (
    SelfPlayEpisodeGenerator,
    TrainingDataExtractor,
    MathProblemGenerator,
    CodeGenerationTaskGenerator
)

async def main():
    # Initialize episode generator
    episode_gen = SelfPlayEpisodeGenerator(
        hrm_agent=None,  # Use simplified agent for demo
        trm_agent=None,
        mcts_config={
            "num_simulations": 100,
            "c_puct": 1.25,
        }
    )

    # Generate tasks
    math_gen = MathProblemGenerator()
    tasks = math_gen.generate(num_tasks=1000)

    # Generate episodes
    episodes = []
    for i, task in enumerate(tasks):
        episode = await episode_gen.generate_episode(
            task=task,
            max_steps=50,
            timeout=60.0
        )
        episodes.append(episode)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/1000 episodes")

    # Analyze success rate
    success_count = sum(1 for ep in episodes if ep.outcome == "success")
    print(f"\nSuccess rate: {success_count / len(episodes):.2%}")

    # Extract training data
    extractor = TrainingDataExtractor()
    training_data = extractor.extract_examples(episodes)

    print(f"\nTraining Examples:")
    print(f"  Policy: {len(training_data['policy'])}")
    print(f"  Value: {len(training_data['value'])}")
    print(f"  Reasoning: {len(training_data['reasoning'])}")
    print(f"  Negative: {len(training_data['negative'])}")

    # Save for training
    # TODO: Save training data

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria:**
- Generate 1,000 episodes
- Success rate >20%
- Extract 5,000+ training examples
- Average episode length <20 steps

**Deliverable:** Training dataset with episode statistics

---

## Session 2: Training Loop Implementation (4 hours)

### Pre-Reading (30 minutes)

- AlphaZero training details
- Policy and value network architectures
- Replay buffer management

### Lecture: AlphaZero Training Implementation (90 minutes)

#### Complete Training Loop

**Full Implementation:**
```python
class SelfPlayTrainer:
    """AlphaZero-style iterative self-play trainer."""

    async def iteration(self, iteration_num: int) -> dict[str, Any]:
        """Run one complete training iteration."""
        metrics = {"iteration": iteration_num}

        # 1. Generate episodes with current model
        num_games = self.config.get("games_per_iteration", 1000)
        episodes = await self.run_self_play(num_games=num_games)
        metrics["num_episodes"] = len(episodes)
        metrics["success_rate"] = (
            sum(1 for ep in episodes if ep.outcome == "success") / len(episodes)
        )

        # 2. Extract training examples
        training_data = self.data_extractor.extract_examples(episodes)
        metrics["num_policy_examples"] = len(training_data["policy"])
        metrics["num_value_examples"] = len(training_data["value"])

        # 3. Train improved model
        if self.hrm_agent is not None:
            train_metrics = await self.train(training_data, iteration_num)
            metrics.update(train_metrics)

        # 4. Evaluate: new vs old
        eval_metrics = await self.evaluate_models(num_eval_games=100)
        metrics.update(eval_metrics)

        # 5. Update best model if improved
        current_metric = eval_metrics.get("eval_success_rate", 0.0)
        if current_metric > self.best_model_metric:
            self.best_model_metric = current_metric
            self._save_checkpoint(iteration_num, best=True)

        return metrics
```

#### Parallel Episode Generation

**Efficient Batching:**
```python
async def run_self_play(self, num_games: int = 1000) -> list[SelfPlayEpisode]:
    """Generate self-play episodes in parallel."""
    # Generate tasks
    tasks = self._generate_tasks(num_games)

    # Generate episodes in batches
    batch_size = self.config.get("parallel_batch_size", 32)
    episodes = []

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i : i + batch_size]

        # Create async tasks for parallel execution
        episode_futures = [
            self.episode_generator.generate_episode(task)
            for task in batch_tasks
        ]

        # Wait for batch to complete
        batch_episodes = await asyncio.gather(*episode_futures)
        episodes.extend(batch_episodes)

        logger.info(f"Generated {len(episodes)}/{num_games} episodes")

    return episodes
```

#### Replay Buffer Management

**Experience Replay:**
```python
class SelfPlayTrainer:
    def __init__(self, ...):
        # ...
        self.episode_buffer = []
        self.max_buffer_size = config.get("max_buffer_size", 10000)

    async def run_self_play(self, num_games: int) -> list[SelfPlayEpisode]:
        """Generate episodes and add to replay buffer."""
        episodes = await self._generate_episodes(num_games)

        # Add to buffer
        self.episode_buffer.extend(episodes)

        # Trim to max size (keep most recent)
        if len(self.episode_buffer) > self.max_buffer_size:
            self.episode_buffer = self.episode_buffer[-self.max_buffer_size:]

        return episodes

    async def train(self, training_data, iteration_num):
        """Train on recent examples + replay buffer."""
        # Use recent examples + samples from buffer
        policy_examples = training_data["policy"]

        # Sample from buffer for experience replay
        if len(self.episode_buffer) > 100:
            buffer_episodes = random.sample(self.episode_buffer, 100)
            buffer_data = self.data_extractor.extract_examples(buffer_episodes)
            policy_examples.extend(buffer_data["policy"])

        # Train on combined dataset
        # ...
```

### Lecture: Model Evaluation (60 minutes)

#### Evaluation Metrics

**Key Metrics:**
1. **Success Rate:** % of episodes that solve task
2. **Average Reward:** Mean cumulative reward
3. **Episode Length:** Average steps to solution
4. **Search Efficiency:** reward / (simulations × time)

**Implementation:**
```python
async def evaluate_models(self, num_eval_games: int = 100) -> dict[str, Any]:
    """Evaluate current models."""
    # Generate evaluation tasks
    eval_tasks = self._generate_tasks(num_eval_games)
    eval_episodes = []

    # Run evaluation episodes
    for task in eval_tasks:
        episode = await self.episode_generator.generate_episode(task)
        eval_episodes.append(episode)

    # Compute metrics
    success_count = sum(1 for ep in eval_episodes if ep.outcome == "success")
    avg_reward = np.mean([ep.total_reward for ep in eval_episodes])
    avg_length = np.mean([ep.episode_length for ep in eval_episodes])
    avg_efficiency = np.mean([ep.search_efficiency for ep in eval_episodes])

    return {
        "eval_success_rate": success_count / len(eval_episodes),
        "eval_avg_reward": float(avg_reward),
        "eval_avg_length": float(avg_length),
        "eval_avg_efficiency": float(avg_efficiency),
    }
```

#### Model Comparison

**New vs Old Model:**
```python
def compare_models(
    old_model,
    new_model,
    num_games: int = 100
) -> dict:
    """Compare two models head-to-head."""
    old_wins = 0
    new_wins = 0
    draws = 0

    for i in range(num_games):
        task = generate_task()

        # Old model attempt
        old_result = old_model.solve(task)

        # New model attempt
        new_result = new_model.solve(task)

        # Compare
        if old_result["reward"] > new_result["reward"]:
            old_wins += 1
        elif new_result["reward"] > old_result["reward"]:
            new_wins += 1
        else:
            draws += 1

    return {
        "old_win_rate": old_wins / num_games,
        "new_win_rate": new_wins / num_games,
        "draw_rate": draws / num_games,
        "recommendation": "deploy_new" if new_wins > old_wins * 1.1 else "keep_old"
    }
```

### Lecture: Checkpointing and Resumability (30 minutes)

**Checkpoint Management:**
```python
def _save_checkpoint(self, iteration: int, best: bool = False) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "best_model_metric": self.best_model_metric,
        "iteration_metrics": self.iteration_metrics,
        "episode_buffer_size": len(self.episode_buffer),
    }

    # Save model states
    if self.hrm_agent is not None:
        checkpoint["hrm_state_dict"] = self.hrm_agent.state_dict()

    if self.trm_agent is not None:
        checkpoint["trm_state_dict"] = self.trm_agent.state_dict()

    # Save checkpoint
    if best:
        path = self.checkpoint_dir / "best_model.pt"
    else:
        path = self.checkpoint_dir / f"iteration_{iteration}.pt"

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(self, path: str) -> None:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=self.device, weights_only=True)

    self.current_iteration = checkpoint["iteration"]
    self.best_model_metric = checkpoint["best_model_metric"]
    self.iteration_metrics = checkpoint["iteration_metrics"]

    if self.hrm_agent is not None and "hrm_state_dict" in checkpoint:
        self.hrm_agent.load_state_dict(checkpoint["hrm_state_dict"])

    logger.info(f"Loaded checkpoint from {path}, iteration {self.current_iteration}")
```

### Hands-On Exercise: Complete Training Loop (60 minutes)

**Exercise 10.2: Run Multi-Iteration Training**

**Objective:** Run 10 iterations of self-play training.

**Requirements:**
1. Initialize trainer with task generators
2. Run 10 training iterations
3. Track metrics across iterations
4. Save best model
5. Generate learning curves

**Deliverable:** Trained model with performance plots

---

## Session 3: RLHF Implementation (4 hours)

### Pre-Reading (30 minutes)

- [RLHF Overview](https://huggingface.co/blog/rlhf)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- Direct Preference Optimization (DPO)

### Lecture: Reinforcement Learning from Human Feedback (90 minutes)

#### RLHF Pipeline

**Three Stages:**
```
1. Supervised Fine-Tuning (SFT)
   - Train on high-quality human demonstrations
   - Creates initial policy π_SFT

2. Reward Model Training
   - Collect human preference data (A vs B)
   - Train reward model: r_θ(x, y)
   - Predicts human preference

3. RL Fine-Tuning (PPO)
   - Optimize policy to maximize reward
   - Policy: π_RL = argmax E[r_θ(x, π(x))]
   - KL penalty to stay close to π_SFT
```

**Simplified Implementation:**
```python
class RLHFTrainer:
    """RLHF training pipeline."""

    def __init__(self, base_model, reward_model, config):
        self.base_model = base_model
        self.reward_model = reward_model
        self.config = config

        # PPO hyperparameters
        self.kl_coef = config.get("kl_coef", 0.1)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)

    def collect_preferences(
        self,
        queries: list[str],
        num_samples_per_query: int = 2
    ) -> list[dict]:
        """Collect human preferences."""
        preferences = []

        for query in queries:
            # Generate multiple responses
            responses = []
            for _ in range(num_samples_per_query):
                response = self.base_model.generate(query)
                responses.append(response)

            # Human labels which response is better
            # In practice, this would be a UI for human annotators
            preference = {
                "query": query,
                "response_a": responses[0],
                "response_b": responses[1],
                "preferred": "a" or "b",  # Human choice
            }
            preferences.append(preference)

        return preferences

    def train_reward_model(
        self,
        preferences: list[dict],
        num_epochs: int = 3
    ):
        """Train reward model on preference data."""
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for pref in preferences:
                # Encode responses
                reward_a = self.reward_model(pref["response_a"])
                reward_b = self.reward_model(pref["response_b"])

                # Bradley-Terry model for preference probability
                # P(a > b) = exp(r_a) / (exp(r_a) + exp(r_b))
                if pref["preferred"] == "a":
                    loss = -torch.log(torch.sigmoid(reward_a - reward_b))
                else:
                    loss = -torch.log(torch.sigmoid(reward_b - reward_a))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss = {total_loss / len(preferences):.4f}")

    def ppo_step(self, query: str, old_logprobs: torch.Tensor):
        """Single PPO optimization step."""
        # Generate response with current policy
        response, new_logprobs = self.base_model.generate_with_logprobs(query)

        # Compute reward
        reward = self.reward_model(response)

        # Compute KL divergence from reference policy
        kl_div = new_logprobs - old_logprobs

        # PPO objective with KL penalty
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        policy_loss = -torch.min(
            ratio * reward,
            clipped_ratio * reward
        )

        # Total loss includes KL penalty
        loss = policy_loss + self.kl_coef * kl_div

        return loss
```

### Lecture: Direct Preference Optimization (DPO) (60 minutes)

**DPO: Simpler Alternative to RLHF**

**Key Insight:** Skip explicit reward model, optimize directly from preferences

**DPO Loss:**
```python
def dpo_loss(model, ref_model, preferences):
    """
    Direct Preference Optimization loss.

    Loss = -E[log σ(β log(π_θ(y_w|x) / π_ref(y_w|x))
                    - β log(π_θ(y_l|x) / π_ref(y_l|x)))]

    where:
        y_w = preferred (winning) response
        y_l = rejected (losing) response
        β = temperature parameter
        σ = sigmoid function
    """
    beta = 0.1

    for pref in preferences:
        # Log probabilities from current model
        log_prob_win = model.log_prob(pref["query"], pref["preferred"])
        log_prob_lose = model.log_prob(pref["query"], pref["rejected"])

        # Log probabilities from reference model
        ref_log_prob_win = ref_model.log_prob(pref["query"], pref["preferred"])
        ref_log_prob_lose = ref_model.log_prob(pref["query"], pref["rejected"])

        # DPO loss
        loss = -torch.log(torch.sigmoid(
            beta * (log_prob_win - ref_log_prob_win) -
            beta * (log_prob_lose - ref_log_prob_lose)
        ))

        yield loss
```

**Advantages of DPO:**
- No separate reward model
- More stable training
- Simpler to implement
- Lower computational cost

### Hands-On Exercise: Implement RLHF (60 minutes)

**Exercise 10.3: Build Preference Dataset**

**Objective:** Collect preference data and train reward model.

**Requirements:**
1. Generate response pairs for 100 queries
2. Simulate preference collection
3. Train reward model
4. Evaluate reward model accuracy

**Deliverable:** Trained reward model with evaluation metrics

---

## Session 4: Production Monitoring & Deployment (4 hours)

### Pre-Reading (30 minutes)

- [training/benchmark_suite.py](../../training/benchmark_suite.py)
- Production ML monitoring best practices

### Lecture: Performance Monitoring (90 minutes)

#### Key Metrics Dashboard

**Metrics to Track:**

**1. Task Performance:**
```python
from prometheus_client import Histogram, Counter, Gauge

# Success metrics
success_rate = Gauge(
    'self_play_success_rate',
    'Success rate of self-play episodes',
    ['task_type']
)

task_completion_time = Histogram(
    'task_completion_seconds',
    'Time to complete tasks',
    ['task_type'],
    buckets=[1, 5, 10, 30, 60, 120]
)

# Quality metrics
solution_quality = Histogram(
    'solution_quality_score',
    'Quality score of solutions',
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# MCTS metrics
mcts_iterations = Histogram(
    'mcts_iterations_count',
    'Number of MCTS iterations',
    buckets=[10, 50, 100, 200, 500, 1000]
)
```

**2. Model Performance:**
```python
# Model metrics
model_inference_time = Histogram(
    'model_inference_seconds',
    'Model inference latency',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

policy_entropy = Histogram(
    'policy_entropy',
    'Policy entropy (exploration measure)',
    buckets=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
)

value_prediction_error = Histogram(
    'value_prediction_error',
    'Absolute error in value prediction',
    buckets=[0.0, 0.1, 0.2, 0.5, 1.0]
)
```

**3. Resource Usage:**
```python
# System metrics
gpu_memory_usage = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage'
)

episode_buffer_size = Gauge(
    'episode_buffer_size',
    'Number of episodes in replay buffer'
)

training_iteration = Counter(
    'training_iterations_total',
    'Total training iterations completed'
)
```

### Lecture: A/B Testing for Model Updates (60 minutes)

**Production A/B Test:**
```python
from training.continual_learning import ABTestFramework

# Initialize A/B test
ab_test = ABTestFramework({
    "traffic_split": 0.1,  # 10% to new model
    "min_samples": 1000,
    "confidence_level": 0.95,
})

# Create test
test_id = ab_test.create_test(
    test_name="model_v2_test",
    model_a=current_model,  # Control
    model_b=new_model,      # Treatment
    metric_fn=lambda inp, out: compute_success_metric(inp, out)
)

# Production traffic
for request in production_requests:
    # Assign to group
    group = ab_test.assign_group(test_id, request.id)

    # Use appropriate model
    model = current_model if group == "A" else new_model
    response = model.solve(request.task)

    # Compute success metric
    success = evaluate_solution(request.task, response)

    # Record result
    ab_test.record_result(
        test_id=test_id,
        group=group,
        input_data=request.task,
        output=response,
        success_metric=success
    )

# Analyze results
status = ab_test.get_test_status(test_id)
if status["status"] == "analyzed":
    result = status["result"]
    print(f"Recommendation: {result['recommendation']}")
    print(f"Improvement: {result['improvement']:.2%}")
    print(f"Statistical significance: {result['is_significant']}")

    if result["recommendation"] == "Deploy B" and result["is_significant"]:
        # Deploy new model
        deploy_model(new_model)
```

### Lecture: Quality Metrics Over Time (60 minutes)

**Tracking Training Progress:**
```python
def get_quality_metrics(trainer: SelfPlayTrainer) -> dict:
    """Compute quality metrics over recent iterations."""
    if not trainer.iteration_metrics:
        return {}

    recent = trainer.iteration_metrics[-10:]  # Last 10 iterations

    # Success rate trend
    success_rates = [m["success_rate"] for m in recent]

    # Compute trend (improvement over iterations)
    if len(success_rates) >= 10:
        early_mean = np.mean(success_rates[:5])
        late_mean = np.mean(success_rates[-5:])
        trend = (late_mean - early_mean) / early_mean if early_mean > 0 else 0
    else:
        trend = 0.0

    return {
        "avg_success_rate": float(np.mean(success_rates)),
        "success_rate_std": float(np.std(success_rates)),
        "success_rate_trend": float(trend),
        "best_success_rate": trainer.best_model_metric,
        "total_iterations": len(trainer.iteration_metrics),
    }
```

**Visualization:**
```python
import matplotlib.pyplot as plt

def plot_training_progress(metrics: list[dict]):
    """Plot training metrics over iterations."""
    iterations = [m["iteration"] for m in metrics]
    success_rates = [m["success_rate"] for m in metrics]
    eval_rates = [m["eval_success_rate"] for m in metrics]

    plt.figure(figsize=(12, 6))

    # Success rate
    plt.subplot(1, 2, 1)
    plt.plot(iterations, success_rates, label="Training", marker='o')
    plt.plot(iterations, eval_rates, label="Evaluation", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Over Training")
    plt.legend()
    plt.grid(True)

    # Episode length
    plt.subplot(1, 2, 2)
    episode_lengths = [m["eval_avg_length"] for m in metrics]
    plt.plot(iterations, episode_lengths, marker='o', color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Average Episode Length")
    plt.title("Solution Efficiency")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()
```

### Hands-On Exercise: Build Monitoring Dashboard (60 minutes)

**Exercise 10.4: Complete Monitoring System**

**Objective:** Deploy monitoring dashboard for self-play training.

**Requirements:**
1. Prometheus metrics collection
2. Grafana dashboard
3. A/B testing framework
4. Alert rules for degradation
5. Training progress visualization

**Deliverable:** Complete monitoring dashboard with alerts

---

## Module 10 Capstone Project

### Final Project: Build Self-Improving System (8 hours)

**Objective:** Build complete end-to-end self-improving AI system.

**Requirements:**

**1. Self-Play Training (30 points)**
- Task generation (math, code, reasoning)
- Episode generation (1,000+ episodes)
- Training data extraction
- Model training loop (5+ iterations)

**2. Evaluation & Testing (25 points)**
- Benchmark suite
- Model comparison
- A/B testing framework
- Statistical significance testing

**3. Production Deployment (25 points)**
- FastAPI deployment
- Monitoring dashboard
- Performance metrics
- Resource tracking

**4. Continual Improvement (20 points)**
- Feedback collection
- Incremental updates
- Quality metrics tracking
- Automated retraining triggers

**Deliverable:**
- GitHub repository with complete code
- Training report with metrics
- Demo video (5-10 minutes)
- Documentation

**Minimum Passing:** 70/100

---

## Assessment Rubric

| Component | Excellent (90-100%) | Good (70-89%) | Needs Work (<70%) |
|-----------|-------------------|---------------|------------------|
| **Self-Play** | 1000+ episodes, >30% success | 500+ episodes, >20% success | <500 episodes |
| **Training** | 10+ iterations, clear improvement | 5+ iterations, some improvement | <5 iterations |
| **Evaluation** | Comprehensive metrics, A/B tests | Basic metrics | Minimal evaluation |
| **Deployment** | Full API, monitoring, docs | Working API | Incomplete |

---

## Additional Resources

### Code References
- [training/self_play_generator.py](../../training/self_play_generator.py)
- [training/continual_learning.py](../../training/continual_learning.py)
- [training/benchmark_suite.py](../../training/benchmark_suite.py)

### Reading
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [MuZero Paper](https://arxiv.org/abs/1911.08265)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

### Tools
- Weights & Biases: https://wandb.ai/
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/

---

## Congratulations!

You've completed the advanced training modules and are now equipped to:

✅ Build state-of-the-art RAG systems
✅ Engineer knowledge bases at scale
✅ Implement self-improving AI systems
✅ Deploy production-ready AI with monitoring
✅ Continuously improve systems from feedback

**You are now a LangGraph Multi-Agent MCTS Expert!** 🎉

---

**Training Program Complete!** 🎓

Continue to:
- Contribute to the framework
- Build real-world applications
- Share knowledge with the community
- Stay current with latest research
