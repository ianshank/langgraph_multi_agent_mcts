# Module 2: Agents Deep Dive (HRM, TRM, MCTS)

**Duration:** 10 hours (2 days)
**Format:** Workshop + Component Lab
**Difficulty:** Intermediate
**Prerequisites:** Completed Module 1, basic pytest knowledge

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand each agent's internal logic** and decision-making process
2. **Modify agent behavior** and validate changes with tests
3. **Instrument agents** with comprehensive LangSmith tracing
4. **Debug agent failures** using trace data and test patterns
5. **Tune agent parameters** for optimal performance

---

## Session 1: HRM Agent (High-Level Reasoning Module) (3 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - HRM section
- [src/agents/hrm_agent.py](../../src/agents/hrm_agent.py) - Implementation
- [tests/components/test_hrm_agent_traced.py](../../tests/components/test_hrm_agent_traced.py) - Test patterns

### Lecture: HRM Internals (60 minutes)

#### Purpose and Responsibilities

**Primary Role:** Decompose complex queries into structured, actionable subtasks

**Key Capabilities:**
1. **Intent Analysis:** Understand the user's high-level goal
2. **Task Decomposition:** Break complex queries into manageable steps
3. **Objective Identification:** Define measurable success criteria
4. **Confidence Calibration:** Estimate decomposition quality

#### Architecture

```python
class HRMAgent:
    def __init__(self, llm, config):
        self.llm = llm
        self.prompt_template = self._load_prompt()
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

    def decompose(self, query: str, context: Optional[dict] = None) -> dict:
        """
        Decompose query into structured tasks.

        Returns:
            {
                "tasks": List[str],
                "objectives": {
                    "primary": str,
                    "secondary": List[str]
                },
                "confidence": float,
                "metadata": dict
            }
        """
        # 1. Analyze intent
        intent = self._analyze_intent(query, context)

        # 2. Generate task decomposition
        tasks = self._generate_tasks(query, intent)

        # 3. Identify objectives
        objectives = self._identify_objectives(query, tasks)

        # 4. Calculate confidence
        confidence = self._calculate_confidence(tasks, objectives)

        return {
            "tasks": tasks,
            "objectives": objectives,
            "confidence": confidence,
            "metadata": {
                "intent": intent,
                "task_count": len(tasks),
                "timestamp": datetime.now()
            }
        }
```

#### Prompt Engineering for HRM

**Key Principles:**
1. **Specificity:** Clear, concrete tasks (not vague statements)
2. **Actionability:** Each task should be measurable
3. **Order:** Logical dependencies (prerequisites first)
4. **Domain Awareness:** Adapt to query type (tactical vs. cybersecurity)

**Example Prompt Template:**
```python
prompt = """
You are a High-Level Reasoning Module (HRM) that decomposes complex queries.

Query Type: {domain}
User Query: {query}

Your task:
1. Analyze the user's intent and goals
2. Break the query into 3-7 specific, actionable subtasks
3. Order tasks by logical dependencies
4. Define primary and secondary objectives

Output Format (JSON):
{
  "tasks": ["Task 1", "Task 2", ...],
  "objectives": {
    "primary": "Main goal",
    "secondary": ["Supporting goal 1", ...]
  },
  "reasoning": "Brief explanation of decomposition strategy"
}

Guidelines:
- Each task must be concrete and measurable
- Focus on {domain}-specific considerations
- Avoid generic or vague statements
"""
```

#### Confidence Calibration

**How Confidence is Calculated:**
```python
def _calculate_confidence(self, tasks, objectives):
    """Calculate confidence score for decomposition."""
    factors = {
        "task_specificity": self._score_specificity(tasks),
        "logical_order": self._score_order(tasks),
        "coverage": self._score_coverage(tasks, objectives),
        "clarity": self._score_clarity(tasks)
    }

    # Weighted average
    weights = {"task_specificity": 0.3, "logical_order": 0.3,
               "coverage": 0.2, "clarity": 0.2}

    confidence = sum(factors[k] * weights[k] for k in factors)
    return confidence
```

**Confidence Thresholds:**
- **High (0.8-1.0):** Clear, well-structured decomposition
- **Medium (0.6-0.8):** Acceptable, may need refinement
- **Low (<0.6):** Requires human review or retry

### Live Demo: HRM in Action (30 minutes)

**Instructor Demo:**

```python
# Run HRM with tracing
from src.agents.hrm_agent import HRMAgent
from tests.utils.langsmith_tracing import trace_e2e_test

@trace_e2e_test(scenario="hrm_tactical_demo")
def demo_hrm_tactical():
    """Demo HRM with tactical query."""
    agent = HRMAgent(llm=ChatOpenAI(), config={})

    query = "What's the best tactical approach for urban warfare?"
    result = agent.decompose(query, context={"domain": "tactical"})

    print(f"Tasks: {result['tasks']}")
    print(f"Confidence: {result['confidence']}")

    # Check LangSmith trace
    print("View trace: https://smith.langchain.com/")

demo_hrm_tactical()
```

**Expected Output:**
```json
{
  "tasks": [
    "Analyze urban terrain characteristics (buildings, streets, civilians)",
    "Identify force composition requirements (infantry, support, equipment)",
    "Evaluate Rules of Engagement for urban environment",
    "Assess logistics and supply chain needs",
    "Develop communication and coordination protocols"
  ],
  "objectives": {
    "primary": "Develop comprehensive urban warfare tactical plan",
    "secondary": [
      "Minimize collateral damage and civilian casualties",
      "Optimize resource allocation and force effectiveness"
    ]
  },
  "confidence": 0.87
}
```

### Hands-On Exercise: Modify HRM (60 minutes)

**Exercise 1: Add Domain-Aware Decomposition**

**Task:** Implement domain detection and use different strategies for tactical vs. cybersecurity queries.

**Steps:**

1. **Add domain detection function:**
```python
def detect_domain(query: str) -> str:
    """Detect query domain: tactical, cybersecurity, or general."""
    keywords_tactical = ["warfare", "military", "tactical", "combat", "defense"]
    keywords_cyber = ["cyber", "security", "hack", "vulnerability", "threat"]

    query_lower = query.lower()

    if any(kw in query_lower for kw in keywords_tactical):
        return "tactical"
    elif any(kw in query_lower for kw in keywords_cyber):
        return "cybersecurity"
    else:
        return "general"
```

2. **Create domain-specific prompts:**
```python
TACTICAL_PROMPT = """
Focus on:
- Terrain analysis
- Force composition
- Rules of Engagement
- Logistics
"""

CYBERSECURITY_PROMPT = """
Focus on:
- Threat vectors
- Vulnerabilities
- Mitigations
- Incident response
"""
```

3. **Update `decompose` method to use domain:**
```python
def decompose(self, query: str, context: Optional[dict] = None) -> dict:
    domain = context.get("domain") if context else detect_domain(query)

    prompt = self._get_domain_prompt(domain)

    # Rest of decomposition logic...
```

4. **Write tests:**
```python
def test_hrm_domain_tactical():
    """Test HRM with tactical query."""
    agent = HRMAgent(llm=MockLLM(), config={})
    query = "Urban warfare tactics"

    result = agent.decompose(query)

    assert result["metadata"]["domain"] == "tactical"
    assert "terrain" in str(result["tasks"]).lower()

def test_hrm_domain_cybersecurity():
    """Test HRM with cybersecurity query."""
    agent = HRMAgent(llm=MockLLM(), config={})
    query = "How to prevent SQL injection attacks?"

    result = agent.decompose(query)

    assert result["metadata"]["domain"] == "cybersecurity"
    assert "vulnerability" in str(result["tasks"]).lower()
```

**Deliverable:** Modified `hrm_agent.py` with domain awareness + passing tests

---

## Session 2: TRM Agent (Tactical Refinement Module) (3 hours)

### Pre-Reading (30 minutes)

- [src/agents/trm_agent.py](../../src/agents/trm_agent.py)
- [tests/components/test_trm_agent_traced.py](../../tests/components/test_trm_agent_traced.py)

### Lecture: TRM Internals (60 minutes)

#### Purpose and Responsibilities

**Primary Role:** Iteratively refine solutions through multi-round reasoning and self-critique

**Key Capabilities:**
1. **Initial Generation:** Create baseline solution
2. **Self-Critique:** Identify weaknesses and gaps
3. **Iterative Refinement:** Improve through multiple rounds
4. **Convergence Detection:** Stop when improvements plateau
5. **Alternative Ranking:** Score and rank multiple candidates

#### Refinement Loop Architecture

```python
class TRMAgent:
    def __init__(self, llm, config):
        self.llm = llm
        self.max_iterations = config.get("max_iterations", 5)
        self.convergence_threshold = config.get("convergence_threshold", 0.05)

    def refine(self, task: str, initial_solution: Optional[str] = None) -> dict:
        """
        Refine solution through iterative improvement.

        Returns:
            {
                "solution": str,
                "alternatives": List[str],
                "iterations": int,
                "converged": bool,
                "improvement_history": List[float]
            }
        """
        # Initialize
        solution = initial_solution or self._generate_initial(task)
        improvements = []

        for i in range(self.max_iterations):
            # Critique current solution
            critique = self._critique(solution, task)

            # Refine based on critique
            new_solution = self._refine_step(solution, critique, task)

            # Measure improvement
            improvement = self._measure_improvement(solution, new_solution)
            improvements.append(improvement)

            # Check convergence
            if improvement < self.convergence_threshold:
                return {
                    "solution": new_solution,
                    "iterations": i + 1,
                    "converged": True,
                    "improvement_history": improvements
                }

            solution = new_solution

        # Max iterations reached
        return {
            "solution": solution,
            "iterations": self.max_iterations,
            "converged": False,
            "improvement_history": improvements
        }
```

#### Critique Prompt Engineering

**Effective Critique Prompt:**
```python
critique_prompt = """
Analyze this solution critically and identify specific areas for improvement.

Task: {task}
Current Solution:
{solution}

Provide a detailed critique addressing:
1. **Missing Information:** What key details are absent?
2. **Logical Gaps:** Are there flawed assumptions or reasoning errors?
3. **Clarity Issues:** Is anything unclear or ambiguous?
4. **Factual Errors:** Are there inaccuracies that need correction?

For each issue, provide:
- Severity (Critical, Important, Minor)
- Specific location in solution
- Concrete suggestion for improvement

Output Format (JSON):
{
  "issues": [
    {
      "type": "missing_info",
      "severity": "critical",
      "description": "...",
      "suggestion": "..."
    },
    ...
  ],
  "overall_quality": 0.0-1.0
}
"""
```

#### Convergence Detection Strategies

**Strategy 1: Improvement Threshold**
```python
def check_convergence_improvement(improvements):
    """Converged if improvement < threshold."""
    return improvements[-1] < 0.05  # 5% improvement
```

**Strategy 2: Moving Average**
```python
def check_convergence_moving_avg(improvements, window=3):
    """Converged if avg improvement over window < threshold."""
    if len(improvements) < window:
        return False
    recent_avg = sum(improvements[-window:]) / window
    return recent_avg < 0.05
```

**Strategy 3: Quality Plateau**
```python
def check_convergence_plateau(quality_scores, window=2):
    """Converged if quality score hasn't changed significantly."""
    if len(quality_scores) < window + 1:
        return False
    recent = quality_scores[-window:]
    variance = max(recent) - min(recent)
    return variance < 0.03  # 3% variance
```

### Live Demo: TRM Refinement (30 minutes)

**Instructor Demo:**
```python
@trace_e2e_test(scenario="trm_refinement_demo")
def demo_trm_refinement():
    """Demo TRM iterative refinement."""
    agent = TRMAgent(llm=ChatOpenAI(), config={
        "max_iterations": 5,
        "convergence_threshold": 0.05
    })

    task = "Develop a strategy to mitigate SQL injection vulnerabilities"
    initial = "Use parameterized queries to prevent SQL injection."

    result = agent.refine(task, initial_solution=initial)

    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    print(f"Improvements: {result['improvement_history']}")
    print(f"Final Solution:\n{result['solution']}")

demo_trm_refinement()
```

**Expected Output:**
```
Iterations: 3
Converged: True
Improvements: [0.15, 0.08, 0.02]
Final Solution:
Comprehensive SQL Injection Mitigation Strategy:

1. Input Validation & Sanitization:
   - Implement strict whitelist validation for all user inputs
   - Use prepared statements with parameterized queries (NEVER string concatenation)
   - Escape special characters for database queries

2. Least Privilege Principle:
   - Database accounts should have minimal required permissions
   - Separate read-only and write accounts
   - Avoid using database admin accounts for web applications

3. Web Application Firewall (WAF):
   - Deploy WAF with SQL injection detection rules
   - Regular expression patterns to identify malicious SQL patterns
   - Log and alert on suspicious activity

4. Code Review & Testing:
   - Automated SAST (Static Application Security Testing)
   - Manual code review focusing on database interaction points
   - Penetration testing with SQLMap and similar tools

5. Monitoring & Incident Response:
   - Log all database queries with source identification
   - Set up alerts for unusual query patterns
   - Incident response plan for detected attacks
```

### Hands-On Exercise: Tune TRM Parameters (60 minutes)

**Exercise 2: Experiment with Refinement Configurations**

**Objective:** Find optimal TRM configuration for quality vs. latency tradeoff

**Steps:**

1. **Set up experiment:**
```python
configs = [
    {"max_iterations": 3, "convergence_threshold": 0.10},
    {"max_iterations": 5, "convergence_threshold": 0.05},
    {"max_iterations": 10, "convergence_threshold": 0.02},
]

results = []
for config in configs:
    agent = TRMAgent(llm=ChatOpenAI(), config=config)

    start = time.time()
    result = agent.refine(task)
    latency = time.time() - start

    results.append({
        "config": config,
        "iterations": result["iterations"],
        "converged": result["converged"],
        "latency": latency,
        "quality": subjective_quality_score(result["solution"])
    })
```

2. **Analyze results:**
```python
import pandas as pd

df = pd.DataFrame(results)
print(df)

# Find sweet spot
df["quality_per_second"] = df["quality"] / df["latency"]
optimal = df.loc[df["quality_per_second"].idxmax()]
print(f"Optimal config: {optimal['config']}")
```

3. **Create visualization:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df["latency"], df["quality"], s=100)
for i, row in df.iterrows():
    plt.annotate(f"iter={row['iterations']}",
                 (row["latency"], row["quality"]))
plt.xlabel("Latency (seconds)")
plt.ylabel("Quality Score")
plt.title("TRM: Quality vs. Latency Tradeoff")
plt.grid(True)
plt.savefig("trm_tuning_results.png")
```

**Deliverable:** Report with optimal configuration recommendation

---

## Session 3: MCTS Agent (Monte Carlo Tree Search) (3 hours)

### Pre-Reading (30 minutes)

- [src/framework/mcts/](../../src/framework/mcts/) - MCTS implementation
- [tests/components/test_mcts_agent_traced.py](../../tests/components/test_mcts_agent_traced.py)
- [examples/mcts_determinism_demo.py](../../examples/mcts_determinism_demo.py)

### Lecture: MCTS Internals (60 minutes)

#### Purpose and Responsibilities

**Primary Role:** Explore decision space using tree search and estimate outcome probabilities

**Key Capabilities:**
1. **Selection:** Choose promising nodes using UCB1 policy
2. **Expansion:** Add new child nodes to explore
3. **Simulation:** Rollout to estimate node value
4. **Backpropagation:** Update ancestors with results

#### MCTS Algorithm

```python
class MCTSEngine:
    def __init__(self, exploration_constant=1.41):
        self.c = exploration_constant  # UCB1 exploration constant

    def search(self, root_state, iterations=100):
        """
        Run MCTS search for specified iterations.

        Returns:
            {
                "best_path": List[Action],
                "win_probability": float,
                "visit_counts": dict,
                "tree": MCTSNode
            }
        """
        root = MCTSNode(state=root_state)

        for i in range(iterations):
            # 1. Selection: Traverse tree using UCB1
            node = self._select(root)

            # 2. Expansion: Add child if not terminal
            if not node.is_terminal() and node.is_fully_expanded():
                node = self._expand(node)

            # 3. Simulation: Rollout to estimate value
            reward = self._simulate(node.state)

            # 4. Backpropagation: Update ancestors
            self._backpropagate(node, reward)

        # Return best action
        best_child = max(root.children, key=lambda n: n.visit_count)
        return {
            "best_path": self._extract_path(best_child),
            "win_probability": best_child.win_rate,
            "visit_counts": {c.action: c.visit_count for c in root.children},
            "tree": root
        }

    def _select(self, node):
        """Select node using UCB1 policy."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = self._best_ucb1_child(node)
        return node

    def _best_ucb1_child(self, node):
        """Select child with highest UCB1 score."""
        return max(node.children, key=lambda n: self._ucb1_score(n))

    def _ucb1_score(self, node):
        """Calculate UCB1 score: exploitation + exploration."""
        if node.visit_count == 0:
            return float('inf')  # Explore unvisited nodes first

        exploitation = node.win_rate
        exploration = self.c * math.sqrt(
            math.log(node.parent.visit_count) / node.visit_count
        )
        return exploitation + exploration
```

#### UCB1 Explained

**Formula:**
```
UCB1(node) = win_rate + c * sqrt(ln(parent_visits) / node_visits)
             └─exploitation─┘   └────────exploration────────┘
```

**Components:**
- **Exploitation:** `win_rate` - Favor nodes with high success rate
- **Exploration:** `c * sqrt(...)` - Favor less-visited nodes
- **c (exploration constant):** Balance between exploitation and exploration
  - Low c (0.5-1.0): More exploitation (greedy)
  - High c (2.0-3.0): More exploration (adventurous)
  - Default: √2 ≈ 1.41 (theoretical optimum)

**Example:**
```python
# Node A: 10 wins in 20 visits, parent has 100 visits
exploitation_A = 10 / 20 = 0.50
exploration_A = 1.41 * sqrt(ln(100) / 20) = 1.41 * 0.48 = 0.68
UCB1_A = 0.50 + 0.68 = 1.18

# Node B: 15 wins in 50 visits, parent has 100 visits
exploitation_B = 15 / 50 = 0.30
exploration_B = 1.41 * sqrt(ln(100) / 50) = 1.41 * 0.30 = 0.42
UCB1_B = 0.30 + 0.42 = 0.72

# Select Node A (higher UCB1 score)
```

#### Simulation (Rollout) Strategies

**Strategy 1: Random Rollout**
```python
def simulate_random(state):
    """Simulate game with random moves."""
    while not state.is_terminal():
        action = random.choice(state.legal_actions())
        state = state.apply_action(action)
    return state.get_reward()
```

**Strategy 2: Heuristic Rollout**
```python
def simulate_heuristic(state, heuristic_fn):
    """Simulate using heuristic to guide moves."""
    while not state.is_terminal():
        actions = state.legal_actions()
        # Score actions using heuristic
        scores = [heuristic_fn(state, a) for a in actions]
        # Choose best action (or sample proportionally)
        action = actions[np.argmax(scores)]
        state = state.apply_action(action)
    return state.get_reward()
```

**Strategy 3: Neural Rollout (Advanced)**
```python
def simulate_neural(state, policy_network):
    """Simulate using learned policy network."""
    while not state.is_terminal():
        # Get action probabilities from network
        probs = policy_network.predict(state.to_tensor())
        action = np.random.choice(state.legal_actions(), p=probs)
        state = state.apply_action(action)
    return state.get_reward()
```

### Live Demo: MCTS Tree Search (30 minutes)

**Instructor Demo:**
```python
@trace_e2e_test(scenario="mcts_tree_search_demo")
def demo_mcts():
    """Demo MCTS tree search."""
    # Simple game state (e.g., Tic-Tac-Toe)
    initial_state = TicTacToeState()

    engine = MCTSEngine(exploration_constant=1.41)
    result = engine.search(initial_state, iterations=100)

    print(f"Best move: {result['best_path'][0]}")
    print(f"Win probability: {result['win_probability']:.2%}")
    print(f"Visit distribution: {result['visit_counts']}")

    # Visualize tree
    visualize_mcts_tree(result['tree'])

demo_mcts()
```

### Hands-On Exercise: Debug MCTS (60 minutes)

**Exercise 3: Fix Suboptimal MCTS Behavior**

**Scenario:** MCTS is selecting clearly suboptimal moves in a game.

**Steps:**

1. **Create failing test:**
```python
def test_mcts_optimal_move():
    """Test that MCTS finds optimal move in obvious position."""
    # Position where one move wins, others lose
    state = create_obvious_winning_position()

    engine = MCTSEngine(exploration_constant=1.41)
    result = engine.search(state, iterations=100)

    # Should choose the winning move
    assert result['best_path'][0] == WINNING_MOVE, \
        f"Expected {WINNING_MOVE}, got {result['best_path'][0]}"
```

2. **Run and observe failure:**
```bash
pytest tests/components/test_mcts_agent_traced.py::test_mcts_optimal_move -v
```

3. **Analyze trace in LangSmith:**
   - Check visit counts: Are all moves explored?
   - Check win rates: Is the winning move's win rate highest?
   - Check UCB1 scores: Is selection working correctly?

4. **Common root causes:**
   - Insufficient iterations
   - Exploration constant too high (over-exploring)
   - Biased simulation rollouts
   - Incorrect backpropagation

5. **Fix and verify:**
```python
# Fix example: Increase iterations
engine = MCTSEngine(exploration_constant=1.41)
result = engine.search(state, iterations=200)  # Increased from 100

# Or: Tune exploration constant
engine = MCTSEngine(exploration_constant=1.0)  # Reduced from 1.41
```

**Deliverable:** Debug report with root cause analysis and fix

---

## Session 4: Integration and Best Practices (1 hour)

### Agent Coordination

**How Agents Work Together:**
```
User Query
    ↓
┌─────────┐
│   HRM   │ → Decompose into tasks
└────┬────┘
     ↓
┌─────────┐
│   TRM   │ → Refine each task iteratively
└────┬────┘
     ↓
┌─────────┐
│  MCTS   │ → Optimize decisions (if applicable)
└─────────┘
     ↓
Final Answer
```

### Best Practices

**1. When to Use Each Agent:**

- **HRM-only:** Simple queries requiring just decomposition
- **HRM + TRM:** Complex queries needing refinement
- **HRM + TRM + MCTS:** Decision-heavy queries with tradeoffs

**2. Performance Optimization:**

- **HRM:** Cache decompositions for similar queries
- **TRM:** Set max_iterations to avoid excessive refinement
- **MCTS:** Balance iterations with latency requirements

**3. Tracing and Debugging:**

- **Always trace** agent execution in development
- **Use metadata** to track performance metrics
- **Analyze failures** in LangSmith before fixing code

**4. Testing Strategy:**

- **Unit tests:** Test individual agent methods
- **Component tests:** Test full agent workflows with mocks
- **E2E tests:** Test agent integration in real scenarios

---

## Module 2 Assessment

### Practical Assessment

**Task:** Modify one agent to add a new capability

**Options:**
1. **HRM:** Domain-aware decomposition (tactical vs. cybersecurity)
2. **TRM:** Adaptive convergence (dynamic threshold based on quality)
3. **MCTS:** Neural guidance (use learned policy for simulation)

**Requirements:**
- Functional implementation (40 points)
- Updated tests with 90%+ coverage (20 points)
- LangSmith traces showing new behavior (10 points)
- Code quality (type hints, formatting) (20 points)
- Documentation (10 points)

**Total:** 100 points (passing: 70+)

**Submission:** Git branch with changes

---

## Additional Resources

### Reading
- [DEEPMIND_IMPLEMENTATION.md](../DEEPMIND_IMPLEMENTATION.md) - Advanced agent patterns
- [examples/deepmind_style_training.py](../../examples/deepmind_style_training.py) - Training loop

### Videos
- Agent architecture walkthrough (to be recorded)
- MCTS visualization tutorial (to be recorded)

### Office Hours
- When: [Schedule TBD]
- Topics: Agent debugging, parameter tuning, test strategies

---

## Next Module

Continue to [MODULE_3_E2E_FLOWS.md](MODULE_3_E2E_FLOWS.md) - E2E Flows & LangGraph Orchestration

**Prerequisites for Module 3:**
- Completed Module 2 practical assessment
- Familiar with LangGraph basics from Module 1
- Async/await understanding (will be covered in depth in Module 6)
