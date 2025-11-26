# Assembly Theory Integration

## Overview

This document describes the integration of **Assembly Theory** into the LangGraph Multi-Agent MCTS framework. Assembly Theory provides a principled way to quantify complexity and guide agent decisions based on construction pathways and reusability.

## What is Assembly Theory?

Assembly Theory is a framework for quantifying the complexity of objects based on the minimum number of steps required to construct them from elementary parts. Key concepts:

- **Assembly Index (A)**: The minimum number of steps needed to construct an object
- **Copy Number (N)**: The frequency with which substructures are reused
- **Assembly Space**: The space of all possible construction pathways

### Key Principles

1. **Complexity as Construction**: Complexity is measured by construction steps, not just size
2. **Reuse Matters**: Objects with high copy numbers are efficiently constructible
3. **Hierarchical Structure**: Complex objects are built from simpler substructures
4. **Selection Principle**: Frequently occurring objects have low assembly indices relative to their copy numbers

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│         Assembly Theory Framework                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌───────────────────┐         │
│  │  Calculator      │    │  Concept          │         │
│  │  - Assembly Index│    │  Extractor        │         │
│  │  - Copy Number   │    │  - NLP            │         │
│  └──────────────────┘    │  - Dependencies   │         │
│                          └───────────────────┘         │
│  ┌──────────────────┐    ┌───────────────────┐         │
│  │  Assembly Graph  │    │  Substructure     │         │
│  │  - Pathways      │    │  Library          │         │
│  │  - Similarity    │    │  - Pattern Reuse  │         │
│  └──────────────────┘    └───────────────────┘         │
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │  Feature Extractor                       │           │
│  │  - 8 Assembly Features for ML Models     │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/framework/assembly/
├── __init__.py                    # Main exports
├── calculator.py                  # Story 1.1: Assembly Index Calculator
├── concept_extractor.py           # Story 1.2: Concept Extraction & Dependencies
├── graph.py                       # Story 1.3: Assembly Graph Data Structure
├── substructure_library.py        # Story 1.4: Pattern Reuse Tracking
├── features.py                    # Story 2.1: Feature Extraction for ML
└── config.py                      # Configuration Management
```

## Usage

### Basic Assembly Index Calculation

```python
from src.framework.assembly import AssemblyIndexCalculator

calculator = AssemblyIndexCalculator()

# Calculate for a query
assembly_index, copy_number = calculator.calculate("How to optimize database queries?")

print(f"Assembly Index: {assembly_index}")  # e.g., 5
print(f"Copy Number: {copy_number}")        # e.g., 2.3
```

### Concept Extraction

```python
from src.framework.assembly import ConceptExtractor

extractor = ConceptExtractor(domain="software")

# Extract concepts
concepts = extractor.extract_concepts("Build REST API with database")

# Build dependency graph
dependency_graph = extractor.build_dependency_graph(concepts)

# Visualize (optional)
extractor.visualize_graph(dependency_graph, "concepts.png")
```

### Assembly Feature Extraction

```python
from src.framework.assembly import AssemblyFeatureExtractor

extractor = AssemblyFeatureExtractor(domain="software")

# Extract all features
features = extractor.extract("Design microservices architecture")

# Access features
print(f"Assembly Index: {features.assembly_index}")
print(f"Decomposability: {features.decomposability_score}")
print(f"Graph Depth: {features.graph_depth}")

# Get human-readable explanation
explanation = extractor.explain_features(features)
print(explanation)

# Convert to ML-ready format
feature_dict = features.to_dict()
feature_array = features.to_array()  # numpy array
```

### Pattern Reuse Tracking

```python
from src.framework.assembly import SubstructureLibrary

library = SubstructureLibrary(max_size=10000)

# Add patterns
sequence = ["state1", "state2", "state3"]
library.add_pattern(sequence, frequency=5)

# Find reusable patterns
matches = library.find_reusable_patterns(["state1", "state2"])

for match in matches:
    print(f"Pattern: {match.sequence}")
    print(f"Frequency: {match.frequency}")
    print(f"Similarity: {match.similarity}")

# Get statistics
stats = library.get_statistics()
print(f"Reuse Rate: {stats['reuse_rate']}")
```

## Integration with Agents

### Meta-Controller Routing

Assembly features enhance routing decisions:

```python
# In meta-controller
from src.framework.assembly import AssemblyFeatureExtractor

feature_extractor = AssemblyFeatureExtractor()
features = feature_extractor.extract(query)

# Route based on assembly index
if features.assembly_index < 3:
    agent = "TRM"  # Simple query
elif features.assembly_index < 7:
    agent = "HRM"  # Medium complexity
else:
    agent = "MCTS"  # High complexity

# Or use decomposability
if features.decomposability_score > 0.7:
    agent = "HRM"  # Highly decomposable → hierarchical reasoning
```

### MCTS Enhancement

Assembly-aware UCB selection:

```python
# In MCTS node selection
def calculate_ucb_with_assembly(node, parent, c=1.41, assembly_weight=0.2):
    base_ucb = node.value/node.visits + c*sqrt(log(parent.visits)/node.visits)

    # Assembly factor: prefer low-complexity paths
    assembly_factor = 1.0 - (node.path_assembly_index / max_index) * assembly_weight

    return base_ucb * assembly_factor
```

### HRM Task Decomposition

Assembly-based hierarchical decomposition:

```python
# In HRM agent
from src.framework.assembly import ConceptExtractor, AssemblyGraph

extractor = ConceptExtractor(domain="software")
concepts = extractor.extract_concepts(query)
dep_graph = extractor.build_dependency_graph(concepts)

# Get assembly layers (topological sort)
layers = list(nx.topological_generations(dep_graph))

# Process layer by layer
for layer_idx, layer_nodes in enumerate(layers):
    print(f"Layer {layer_idx}: {layer_nodes}")
    # Process this assembly layer
```

### TRM Convergence

Assembly-aware convergence detection:

```python
# In TRM iteration
complexity_history = []

for iteration in range(max_iterations):
    answer = trm.refine(query, previous_answer)

    # Calculate assembly index of answer
    answer_complexity = calculator.calculate(answer.reasoning_trace)
    complexity_history.append(answer_complexity[0])

    # Check complexity convergence
    if len(complexity_history) > 1:
        complexity_reduction = complexity_history[-2] - complexity_history[-1]

        if complexity_reduction < threshold:
            print(f"Converged at iteration {iteration}")
            break
```

## Configuration

Assembly behavior is controlled via `config/assembly_config.yaml`:

```yaml
assembly:
  mcts:
    ucb_weight: 0.2                  # Assembly factor weight
    max_complexity_threshold: 15      # Pruning threshold

  routing:
    simple_threshold: 3               # TRM routing threshold
    medium_threshold: 7               # HRM routing threshold

  feature_flags:
    routing: true                     # Enable assembly routing
    mcts: true                        # Enable MCTS enhancements
    hrm: true                         # Enable HRM enhancements
    trm: true                         # Enable TRM enhancements
```

Load configuration:

```python
from src.framework.assembly import AssemblyConfig

config = AssemblyConfig.from_yaml("config/assembly_config.yaml")
```

## Features Extracted

The `AssemblyFeatureExtractor` provides 8 features:

1. **assembly_index**: Core complexity measure (0-20+)
2. **copy_number**: Substructure reuse factor (1.0+)
3. **decomposability_score**: How easily decomposed (0.0-1.0)
4. **graph_depth**: Dependency graph depth (0-10+)
5. **constraint_count**: Number of dependencies
6. **concept_count**: Number of extracted concepts
7. **technical_complexity**: Technical term density (0.0-1.0)
8. **normalized_assembly_index**: Normalized complexity (0.0-1.0)

These features can be used for:
- Meta-controller routing decisions
- MCTS node selection
- HRM decomposition
- TRM convergence
- Consensus scoring

## Performance

### Benchmarks

Target performance (on typical queries):

- Assembly index calculation: < 10ms
- Concept extraction: < 20ms
- Feature extraction (complete): < 50ms
- Pattern library lookup: < 5ms

### Caching

Assembly indices are cached by default:

```python
calculator = AssemblyIndexCalculator(cache_enabled=True, max_cache_size=10000)

# First call: calculates
idx1, cn1 = calculator.calculate(query)  # ~10ms

# Second call: cached
idx2, cn2 = calculator.calculate(query)  # ~0.1ms
```

## Testing

Run the test suite:

```bash
# All assembly tests
pytest tests/framework/assembly/ -v

# Specific component
pytest tests/framework/assembly/test_calculator.py -v

# With coverage
pytest tests/framework/assembly/ --cov=src/framework/assembly
```

## References

1. **Assembly Theory Paper**: "Assembly Theory Explains and Quantifies Selection and Evolution" - Cronin, Walker, et al.
2. **Application to AI**: Complexity-guided search and modular reasoning
3. **LangGraph Documentation**: [langgraph.dev](https://langgraph.dev)

## Future Work

### Sprint 2-8 (Remaining Stories)

- Story 2.2-2.5: Meta-controller integration and retraining
- Story 3.1-3.5: Full MCTS enhancement with pruning
- Story 4.1-4.3: Complete HRM assembly integration
- Story 5.1-5.3: TRM refinement with assembly
- Story 6.1-6.3: Assembly-based consensus
- Story 7.1-7.3: W&B observability dashboard
- Story 8.3-8.4: Comprehensive benchmarks and ablation studies
- Story 9.1-9.4: Complete documentation and examples
- Story 10.1-10.4: Production deployment

## Support

For questions or issues:
- Check this documentation
- Review test cases in `tests/framework/assembly/`
- Examine example code in module docstrings
- Open GitHub issues for bugs or feature requests
