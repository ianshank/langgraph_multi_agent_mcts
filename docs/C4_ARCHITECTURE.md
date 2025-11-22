# C4 Architecture Diagrams: DeepMind-Style Self-Improving AI System

This document provides C4 architecture diagrams (Context, Container, Component, and Code levels) for the LangGraph Multi-Agent MCTS framework with DeepMind-style learning.

## Table of Contents

1. [Level 1: System Context Diagram](#level-1-system-context-diagram)
2. [Level 2: Container Diagram](#level-2-container-diagram)
3. [Level 3: Component Diagrams](#level-3-component-diagrams)
4. [Level 4: Code Diagrams](#level-4-code-diagrams)
5. [Deployment Architecture](#deployment-architecture)
6. [Data Flow Diagrams](#data-flow-diagrams)

---

## Level 1: System Context Diagram

Shows the system in its environment with external actors and systems.

```mermaid
graph TB
    subgraph "External Systems & Users"
        User[AI Researcher/Developer]
        Client[Client Applications]
        WandB[Weights & Biases]
        Storage[Cloud Storage S3]
        Monitor[Monitoring Systems]
    end

    subgraph "LangGraph Multi-Agent MCTS System"
        System[DeepMind-Style<br/>Self-Improving AI System<br/><br/>Combines HRM, TRM, and Neural MCTS<br/>for hierarchical reasoning and<br/>self-improving capabilities]
    end

    User -->|Configure & Train| System
    User -->|Monitor Progress| System
    Client -->|Inference Requests<br/>REST API| System
    System -->|Log Experiments<br/>Metrics| WandB
    System -->|Store Checkpoints<br/>Training Data| Storage
    System -->|Performance Metrics<br/>Telemetry| Monitor
    System -->|Trained Models<br/>Predictions| Client

    style System fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style User fill:#95E1D3,stroke:#38A89D,stroke-width:2px
    style Client fill:#95E1D3,stroke:#38A89D,stroke-width:2px
    style WandB fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style Storage fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style Monitor fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
```

### Key Relationships

| From | To | Description |
|------|-----|-------------|
| **AI Researcher** | System | Configures training parameters, monitors experiments |
| **Client Applications** | System | Makes inference requests via REST API |
| **System** | Weights & Biases | Logs training metrics, experiments, model performance |
| **System** | Cloud Storage | Persists checkpoints, training data, replay buffer |
| **System** | Monitoring | Sends telemetry, performance metrics, alerts |

---

## Level 2: Container Diagram

Shows the high-level technical building blocks (applications, data stores, services).

```mermaid
graph TB
    subgraph "User & Client Layer"
        User[AI Researcher]
        Client[Client App]
    end

    subgraph "LangGraph Multi-Agent MCTS System"
        subgraph "Training System"
            Orchestrator[Training Orchestrator<br/>Python/AsyncIO<br/><br/>Coordinates training pipeline]
            SelfPlay[Self-Play Generator<br/>Python/AsyncIO<br/><br/>Generates training data]
            Monitor[Performance Monitor<br/>Python<br/><br/>Tracks metrics]
        end

        subgraph "Core Models"
            HRM[HRM Agent<br/>PyTorch<br/><br/>Hierarchical reasoning]
            TRM[TRM Agent<br/>PyTorch<br/><br/>Recursive refinement]
            MCTS[Neural MCTS<br/>Python/PyTorch<br/><br/>Tree search with NN]
            ParallelMCTS[Parallel MCTS<br/>AsyncIO/PyTorch<br/><br/>Virtual loss parallelization]
            PVNet[Policy-Value Network<br/>PyTorch ResNet<br/><br/>Action probabilities + value]
            PolicyNet[Policy Network<br/>PyTorch MLP<br/><br/>Fast action selection]
            ValueNet[Value Network<br/>PyTorch MLP<br/><br/>Position evaluation]
            HybridAgent[Hybrid Agent<br/>PyTorch/LLM<br/><br/>Neural + LLM reasoning]
        end

        subgraph "Inference System"
            API[FastAPI Server<br/>FastAPI/Uvicorn<br/><br/>REST API endpoints]
            InferenceEngine[Inference Engine<br/>Python/PyTorch<br/><br/>Model inference]
        end

        subgraph "Data Layer"
            ReplayBuffer[Replay Buffer<br/>Python/NumPy<br/><br/>Experience storage]
            Cache[Evaluation Cache<br/>Python Dict<br/><br/>MCTS caching]
        end
    end

    subgraph "External Services"
        WandB[(Weights & Biases<br/>Experiment Tracking)]
        S3[(Cloud Storage<br/>S3/MinIO)]
        Prometheus[(Monitoring<br/>Prometheus/Grafana)]
    end

    User -->|Configure| Orchestrator
    Client -->|HTTP POST| API

    Orchestrator -->|Orchestrates| SelfPlay
    Orchestrator -->|Trains| HRM
    Orchestrator -->|Trains| TRM
    Orchestrator -->|Trains| PVNet
    Orchestrator -->|Trains| PolicyNet
    Orchestrator -->|Trains| ValueNet
    Orchestrator -->|Uses| Monitor

    SelfPlay -->|Uses| MCTS
    SelfPlay -->|Uses| ParallelMCTS
    SelfPlay -->|Stores| ReplayBuffer

    MCTS -->|Evaluates with| PVNet
    MCTS -->|Caches in| Cache
    MCTS -->|Guided by| HRM
    MCTS -->|Refines with| TRM

    ParallelMCTS -->|Evaluates with| PolicyNet
    ParallelMCTS -->|Evaluates with| ValueNet

    HybridAgent -->|Uses| PolicyNet
    HybridAgent -->|Uses| ValueNet
    HybridAgent -->|Falls back to| HRM

    API -->|Routes to| InferenceEngine
    InferenceEngine -->|Uses| HRM
    InferenceEngine -->|Uses| TRM
    InferenceEngine -->|Uses| MCTS
    InferenceEngine -->|Uses| ParallelMCTS
    InferenceEngine -->|Uses| PVNet
    InferenceEngine -->|Uses| HybridAgent

    Orchestrator -->|Logs to| WandB
    Monitor -->|Exports to| Prometheus
    Orchestrator -->|Saves checkpoints| S3
    ReplayBuffer -->|Persists to| S3

    style Orchestrator fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style SelfPlay fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style HRM fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style TRM fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style MCTS fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style PVNet fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style API fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
    style ReplayBuffer fill:#F39C12,stroke:#B9770E,stroke-width:2px
    style Cache fill:#F39C12,stroke:#B9770E,stroke-width:2px
```

### Container Descriptions

| Container | Technology | Responsibility |
|-----------|-----------|----------------|
| **Training Orchestrator** | Python, PyTorch, AsyncIO | Coordinates complete training pipeline |
| **Self-Play Generator** | Python, AsyncIO | Generates training data through self-play |
| **HRM Agent** | PyTorch, Transformers | Hierarchical problem decomposition |
| **TRM Agent** | PyTorch, GRU | Recursive solution refinement |
| **Neural MCTS** | Python, NumPy, PyTorch | Tree search with neural guidance |
| **Parallel MCTS** | Python, AsyncIO, PyTorch | Parallelized tree search with virtual loss |
| **Policy-Value Network** | PyTorch, ResNet | Predicts action probabilities and values |
| **Policy Network** | PyTorch, MLP | Fast action probability prediction |
| **Value Network** | PyTorch, MLP | Fast position value estimation |
| **Hybrid Agent** | PyTorch, LLM | Cost-optimized neural+LLM reasoning |
| **FastAPI Server** | FastAPI, Uvicorn | REST API for inference |
| **Inference Engine** | PyTorch | Model inference and prediction |
| **Replay Buffer** | Python, NumPy | Stores and samples experiences |
| **Evaluation Cache** | Python Dict, LRU | Caches MCTS evaluations |

---

## Level 3: Component Diagrams

### 3.1 Training Orchestrator Components

```mermaid
graph TB
    subgraph "Training Orchestrator Container"
        subgraph "Core Orchestration"
            OrcMain[UnifiedTrainingOrchestrator<br/><br/>Main coordinator]
            IterMgr[Iteration Manager<br/><br/>Training loop control]
            EvalMgr[Evaluation Manager<br/><br/>Model evaluation]
        end

        subgraph "Model Training"
            PVTrainer[PV Network Trainer<br/><br/>Policy-value training]
            HRMTrainer[HRM Trainer<br/><br/>Hierarchical reasoning]
            TRMTrainer[TRM Trainer<br/><br/>Recursive refinement]
            LossCompute[Loss Functions<br/><br/>AlphaZero, HRM, TRM losses]
        end

        subgraph "Optimizers"
            PVOptim[SGD Optimizer<br/><br/>+ momentum]
            HRMOptim[Adam Optimizer<br/><br/>HRM params]
            TRMOptim[Adam Optimizer<br/><br/>TRM params]
            LRScheduler[LR Scheduler<br/><br/>Cosine annealing]
        end

        subgraph "Data Pipeline"
            SelfPlayCollector[Self-Play Collector<br/><br/>Game generation]
            DataSampler[Batch Sampler<br/><br/>PER sampling]
            Augmenter[Data Augmenter<br/><br/>Symmetries]
        end

        subgraph "Persistence"
            CheckpointMgr[Checkpoint Manager<br/><br/>Save/load models]
            ConfigMgr[Config Manager<br/><br/>System configuration]
        end

        subgraph "Monitoring"
            PerfMon[Performance Monitor<br/><br/>Metrics tracking]
            WandBLogger[WandB Logger<br/><br/>Experiment logging]
            TimingCtx[Timing Context<br/><br/>Profiling]
        end
    end

    OrcMain -->|Manages| IterMgr
    OrcMain -->|Uses| EvalMgr

    IterMgr -->|Phase 1| SelfPlayCollector
    IterMgr -->|Phase 2| PVTrainer
    IterMgr -->|Phase 3| HRMTrainer
    IterMgr -->|Phase 4| TRMTrainer
    IterMgr -->|Phase 5| EvalMgr

    SelfPlayCollector -->|Stores| DataSampler
    DataSampler -->|Applies| Augmenter

    PVTrainer -->|Uses| PVOptim
    PVTrainer -->|Computes| LossCompute
    PVTrainer -->|Samples from| DataSampler

    HRMTrainer -->|Uses| HRMOptim
    TRMTrainer -->|Uses| TRMOptim

    PVOptim -->|Adjusts| LRScheduler

    OrcMain -->|Persists with| CheckpointMgr
    OrcMain -->|Configured by| ConfigMgr

    OrcMain -->|Monitors with| PerfMon
    PerfMon -->|Logs to| WandBLogger
    IterMgr -->|Times with| TimingCtx
    TimingCtx -->|Reports to| PerfMon

    style OrcMain fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
```

### 3.2 Neural MCTS Components

```mermaid
graph TB
    subgraph "Neural MCTS Container"
        subgraph "Core MCTS"
            MCTSEngine[MCTS Engine<br/><br/>Main search controller]
            TreeSearch[Tree Search<br/><br/>PUCT traversal]
            NodeMgr[Node Manager<br/><br/>Tree node operations]
        end

        subgraph "Node Operations"
            Selection[Selection Phase<br/><br/>PUCT selection]
            Expansion[Expansion Phase<br/><br/>Add children]
            Simulation[Simulation Phase<br/><br/>Rollout/evaluate]
            Backprop[Backpropagation<br/><br/>Update ancestors]
        end

        subgraph "Neural Guidance"
            NNEvaluator[NN Evaluator<br/><br/>Policy-value prediction]
            EvalCache[Evaluation Cache<br/><br/>LRU cache]
            DirichletNoise[Dirichlet Noise<br/><br/>Root exploration]
        end

        subgraph "Action Selection"
            VisitCount[Visit Counter<br/><br/>Track visits]
            TempControl[Temperature Control<br/><br/>Exploration decay]
            ActionSampler[Action Sampler<br/><br/>Select action]
        end

        subgraph "Parallel Support"
            VirtualLoss[Virtual Loss<br/><br/>Parallel discourage]
            Semaphore[Async Semaphore<br/><br/>Concurrency control]
        end
    end

    MCTSEngine -->|Executes| TreeSearch
    TreeSearch -->|Manages| NodeMgr

    TreeSearch -->|Step 1| Selection
    Selection -->|Step 2| Expansion
    Expansion -->|Step 3| Simulation
    Simulation -->|Step 4| Backprop

    Selection -->|Uses PUCT| NodeMgr
    Expansion -->|Gets priors from| NNEvaluator
    Simulation -->|Evaluates with| NNEvaluator

    NNEvaluator -->|Checks| EvalCache
    NNEvaluator -->|At root adds| DirichletNoise

    Backprop -->|Updates| VisitCount
    MCTSEngine -->|Applies| TempControl
    TempControl -->|Influences| ActionSampler
    VisitCount -->|Informs| ActionSampler

    Selection -->|Adds| VirtualLoss
    Backprop -->|Removes| VirtualLoss
    TreeSearch -->|Controls with| Semaphore

    style MCTSEngine fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
```

### 3.3 Inference API Components

```mermaid
graph TB
    subgraph "FastAPI Inference Server"
        subgraph "API Layer"
            FastAPIApp[FastAPI Application<br/><br/>Route definitions]
            Endpoints[REST Endpoints<br/><br/>/inference, /policy-value]
            Middleware[CORS Middleware<br/><br/>Cross-origin support]
            Validation[Request Validation<br/><br/>Pydantic models]
        end

        subgraph "Inference Engine"
            InferController[Inference Controller<br/><br/>Request orchestration]
            ModelLoader[Model Loader<br/><br/>Load from checkpoint]
            StateConverter[State Converter<br/><br/>Input preprocessing]
        end

        subgraph "Inference Pipeline"
            HRMInference[HRM Inference<br/><br/>Problem decomposition]
            MCTSInference[MCTS Inference<br/><br/>Tree search]
            TRMInference[TRM Inference<br/><br/>Solution refinement]
            PVInference[PV Inference<br/><br/>Direct evaluation]
        end

        subgraph "Response Processing"
            ResultAggregator[Result Aggregator<br/><br/>Combine outputs]
            ResponseBuilder[Response Builder<br/><br/>Format results]
            ErrorHandler[Error Handler<br/><br/>Exception handling]
        end

        subgraph "Monitoring"
            HealthCheck[Health Check<br/><br/>System status]
            StatsCollector[Stats Collector<br/><br/>Performance metrics]
            MemoryMonitor[Memory Monitor<br/><br/>GPU/CPU tracking]
        end
    end

    FastAPIApp -->|Defines| Endpoints
    FastAPIApp -->|Uses| Middleware
    Endpoints -->|Validates with| Validation

    Endpoints -->|Routes to| InferController
    InferController -->|Loads models via| ModelLoader
    InferController -->|Converts with| StateConverter

    InferController -->|Optional| HRMInference
    InferController -->|Optional| MCTSInference
    InferController -->|Optional| TRMInference
    InferController -->|Direct| PVInference

    HRMInference -->|Feeds| MCTSInference
    MCTSInference -->|Refines with| TRMInference

    HRMInference -->|Results to| ResultAggregator
    MCTSInference -->|Results to| ResultAggregator
    TRMInference -->|Results to| ResultAggregator
    PVInference -->|Results to| ResultAggregator

    ResultAggregator -->|Formats with| ResponseBuilder
    InferController -->|Errors to| ErrorHandler

    Endpoints -->|Health endpoint| HealthCheck
    InferController -->|Tracks with| StatsCollector
    HealthCheck -->|Checks| MemoryMonitor

    style FastAPIApp fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
```

### 3.4 HRM Agent Components

```mermaid
graph TB
    subgraph "HRM Agent Container"
        subgraph "Input Processing"
            InputProj[Input Projection<br/><br/>Linear projection]
            StateEncoder[State Encoder<br/><br/>Initial encoding]
        end

        subgraph "H-Module (High-Level)"
            HAttention[Multi-Head Attention<br/><br/>Relational reasoning]
            HFFN[Feed-Forward Network<br/><br/>4x expansion]
            HNorm[Layer Normalization<br/><br/>Stable training]
            DecompHead[Decomposition Head<br/><br/>Subproblem generation]
        end

        subgraph "L-Module (Low-Level)"
            HtoL[H→L Projection<br/><br/>Dimension mapping]
            LGRU[GRU Layers<br/><br/>Sequential processing]
            LOutput[Output Projection<br/><br/>Result generation]
            LtoH[L→H Projection<br/><br/>Feedback path]
        end

        subgraph "Adaptive Computation"
            ACTUnit[ACT Halting Unit<br/><br/>Confidence prediction]
            PonderCost[Ponder Cost<br/><br/>Efficiency regularization]
            ConvergenceTrack[Convergence Tracker<br/><br/>Path monitoring]
        end

        subgraph "Integration"
            StateIntegrate[State Integration<br/><br/>Combine H+L feedback]
            IterController[Iteration Controller<br/><br/>Loop management]
        end
    end

    InputProj -->|Encodes| StateEncoder
    StateEncoder -->|Feeds| HAttention

    HAttention -->|Through| HFFN
    HFFN -->|Normalized by| HNorm
    HNorm -->|Can generate| DecompHead

    HNorm -->|Projects to| HtoL
    HtoL -->|Processes in| LGRU
    LGRU -->|Outputs via| LOutput
    LOutput -->|Feeds back via| LtoH

    HNorm -->|Evaluated by| ACTUnit
    ACTUnit -->|Computes| PonderCost
    ACTUnit -->|Tracks in| ConvergenceTrack

    LtoH -->|Combined in| StateIntegrate
    HNorm -->|Combined in| StateIntegrate
    StateIntegrate -->|Controlled by| IterController

    ConvergenceTrack -->|Informs| IterController
    IterController -->|May continue| HAttention

    style HAttention fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
```

### 3.5 Advanced MCTS Components (Module 8)

```mermaid
graph TB
    subgraph "Advanced MCTS Container"
        subgraph "Parallel MCTS Engine"
            ParEngine[Parallel MCTS Engine<br/><br/>AsyncIO coordination]
            VirtualLossCtrl[Virtual Loss Controller<br/><br/>Collision prevention]
            WorkerPool[Worker Pool<br/><br/>Concurrent simulations]
        end

        subgraph "Virtual Loss Mechanism"
            VLNode[Virtual Loss Node<br/><br/>Extended MCTS node]
            VLAdd[Add Virtual Loss<br/><br/>Pessimistic value]
            VLRevert[Revert Virtual Loss<br/><br/>Restore true value]
            CollisionTrack[Collision Tracker<br/><br/>Monitor conflicts]
        end

        subgraph "Progressive Widening"
            PWEngine[Progressive Widening Engine<br/><br/>Adaptive expansion]
            PWCriterion[Expansion Criterion<br/><br/>N(s) > k * |C(s)|^α]
            AdaptiveK[Adaptive K Tuner<br/><br/>Variance-based]
            ActionFilter[Action Filter<br/><br/>Prune unpromising]
        end

        subgraph "RAVE (Rapid Action Value Estimation)"
            RAVENode[RAVE Node<br/><br/>AMAF statistics]
            RAVEBackprop[RAVE Backpropagation<br/><br/>All-Moves-As-First]
            BetaDecay[Beta Decay<br/><br/>UCB-RAVE mixing]
            HybridSelect[Hybrid Selection<br/><br/>(1-β)*UCB + β*RAVE]
        end

        subgraph "Parallelization Strategies"
            TreePar[Tree Parallelization<br/><br/>Shared tree + VL]
            RootPar[Root Parallelization<br/><br/>Independent trees]
            LeafPar[Leaf Parallelization<br/><br/>Parallel rollouts]
        end
    end

    ParEngine -->|Manages| WorkerPool
    ParEngine -->|Uses| VirtualLossCtrl

    VirtualLossCtrl -->|Creates| VLNode
    VLNode -->|Adds pessimism| VLAdd
    VLNode -->|Restores value| VLRevert
    VLAdd -->|Tracked by| CollisionTrack

    PWEngine -->|Evaluates| PWCriterion
    PWCriterion -->|Adjusts| AdaptiveK
    PWEngine -->|Applies| ActionFilter

    RAVENode -->|Stores| RAVEBackprop
    RAVEBackprop -->|Computes| BetaDecay
    BetaDecay -->|Influences| HybridSelect

    ParEngine -->|Strategy 1| TreePar
    ParEngine -->|Strategy 2| RootPar
    ParEngine -->|Strategy 3| LeafPar

    style ParEngine fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
    style PWEngine fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
```

### 3.6 Neural Network Components (Module 9)

```mermaid
graph TB
    subgraph "Neural Network Integration Container"
        subgraph "Policy Network"
            PolInput[Input Layer<br/><br/>State encoding]
            PolHidden[Hidden Layers<br/><br/>2-3 layers, 256-512 units]
            PolBatchNorm[Batch Normalization<br/><br/>Training stability]
            PolDropout[Dropout Layer<br/><br/>Regularization]
            PolOutput[Output Layer<br/><br/>Action probabilities]
        end

        subgraph "Value Network"
            ValInput[Input Layer<br/><br/>State encoding]
            ValHidden[Hidden Layers<br/><br/>2-3 layers, 256-512 units]
            ValBatchNorm[Batch Normalization<br/><br/>Training stability]
            ValDropout[Dropout Layer<br/><br/>Regularization]
            ValOutput[Output Layer<br/><br/>Value estimate]
        end

        subgraph "Neural Trainer"
            DataLoader[Data Loader<br/><br/>Batch sampling]
            LossCompute[Loss Computation<br/><br/>Policy + Value loss]
            Optimizer[Adam Optimizer<br/><br/>Learning rate scheduler]
            ValTracker[Validation Tracker<br/><br/>Early stopping]
            CheckpointMgr[Checkpoint Manager<br/><br/>Model versioning]
        end

        subgraph "Data Collection"
            MCTSCollect[MCTS Data Collector<br/><br/>Self-play games]
            StateConvert[State Converter<br/><br/>Format adaptation]
            PolicyTarget[Policy Target<br/><br/>MCTS visit distribution]
            ValueTarget[Value Target<br/><br/>Game outcome]
        end

        subgraph "Hybrid Agent"
            ThresholdCheck[Confidence Threshold<br/><br/>Decision router]
            NeuralPath[Neural Inference Path<br/><br/>Fast, cheap]
            LLMPath[LLM Inference Path<br/><br/>High quality, expensive]
            ResultMerge[Result Merger<br/><br/>Confidence weighting]
            CostTracker[Cost Tracker<br/><br/>ROI monitoring]
        end
    end

    PolInput -->|Through| PolHidden
    PolHidden -->|Normalized by| PolBatchNorm
    PolBatchNorm -->|Regularized by| PolDropout
    PolDropout -->|Outputs| PolOutput

    ValInput -->|Through| ValHidden
    ValHidden -->|Normalized by| ValBatchNorm
    ValBatchNorm -->|Regularized by| ValDropout
    ValDropout -->|Outputs| ValOutput

    DataLoader -->|Feeds| LossCompute
    LossCompute -->|Updates via| Optimizer
    Optimizer -->|Validates with| ValTracker
    ValTracker -->|Saves to| CheckpointMgr

    MCTSCollect -->|Converts via| StateConvert
    StateConvert -->|Generates| PolicyTarget
    StateConvert -->|Generates| ValueTarget
    PolicyTarget -->|Trains| PolOutput
    ValueTarget -->|Trains| ValOutput

    ThresholdCheck -->|High confidence| NeuralPath
    ThresholdCheck -->|Low confidence| LLMPath
    NeuralPath -->|Uses| PolOutput
    NeuralPath -->|Uses| ValOutput
    NeuralPath -->|Result to| ResultMerge
    LLMPath -->|Result to| ResultMerge
    ResultMerge -->|Tracked by| CostTracker

    style PolOutput fill:#E67E22,stroke:#BA4A00,stroke-width:2px,color:#fff
    style ValOutput fill:#E67E22,stroke:#BA4A00,stroke-width:2px,color:#fff
    style ThresholdCheck fill:#16A085,stroke:#0E6655,stroke-width:2px,color:#fff
```

### 3.7 TRM Agent Components

```mermaid
graph TB
    subgraph "TRM Agent Container"
        subgraph "Input Processing"
            Encoder[Initial Encoder<br/><br/>2-layer MLP]
            InputNorm[Input Normalization<br/><br/>Layer norm]
        end

        subgraph "Recursive Processing"
            RecBlock[Recursive Block<br/><br/>Shared weights]
            Transform[Transformation<br/><br/>Hidden → Latent]
            ResidualScale[Residual Scaling<br/><br/>Learned α]
            RecNorm[Block Normalization<br/><br/>Layer norm]
        end

        subgraph "Deep Supervision"
            SuperHead1[Supervision Head 1<br/><br/>Early prediction]
            SuperHead2[Supervision Head 2<br/><br/>Mid prediction]
            SuperHeadN[Supervision Head N<br/><br/>Late prediction]
            OutputHead[Output Head<br/><br/>Final prediction]
        end

        subgraph "Convergence Detection"
            ResidualCalc[Residual Calculator<br/><br/>L2 distance]
            ThresholdCheck[Threshold Check<br/><br/>Early stopping]
            StepCounter[Step Counter<br/><br/>Recursion tracking]
        end

        subgraph "State Management"
            StateClone[State Cloning<br/><br/>Previous state copy]
            DeltaCompute[Delta Computer<br/><br/>State difference]
            IterManager[Iteration Manager<br/><br/>Loop control]
        end
    end

    Encoder -->|Normalizes via| InputNorm
    InputNorm -->|Feeds| RecBlock

    RecBlock -->|Applies| Transform
    Transform -->|Scaled by| ResidualScale
    ResidualScale -->|Normalized by| RecNorm

    RecBlock -->|Iteration 1→| SuperHead1
    RecBlock -->|Iteration i→| SuperHead2
    RecBlock -->|Iteration N→| SuperHeadN
    RecBlock -->|Final→| OutputHead

    RecBlock -->|Clones to| StateClone
    RecNorm -->|Compared with| StateClone
    StateClone -->|Via| DeltaCompute
    DeltaCompute -->|Measures| ResidualCalc

    ResidualCalc -->|Checks| ThresholdCheck
    ThresholdCheck -->|Updates| StepCounter
    StepCounter -->|Controls| IterManager

    ThresholdCheck -->|May stop| RecBlock
    IterManager -->|May continue| RecBlock

    style RecBlock fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
```

---

## Level 4: Code Diagrams

### 4.1 HRM Agent Class Structure

```mermaid
classDiagram
    class HRMConfig {
        +int h_dim
        +int l_dim
        +int num_h_layers
        +int num_l_layers
        +int max_outer_steps
        +float halt_threshold
        +bool use_augmentation
        +float dropout
    }

    class SubProblem {
        +int level
        +str description
        +Tensor state
        +int parent_id
        +float confidence
    }

    class HRMOutput {
        +Tensor final_state
        +List~SubProblem~ subproblems
        +int halt_step
        +float total_ponder_cost
        +List~float~ convergence_path
    }

    class AdaptiveComputationTime {
        +float epsilon
        +Sequential halt_fc
        +forward(hidden_states) Tuple
    }

    class HModule {
        +HRMConfig config
        +MultiheadAttention attention
        +Sequential ffn
        +LayerNorm norm1
        +LayerNorm norm2
        +Sequential decompose_head
        +forward(x) Tensor
        +decompose(x) Tensor
    }

    class LModule {
        +HRMConfig config
        +Linear h_to_l
        +GRU gru
        +Sequential output_proj
        +Linear l_to_h
        +forward(x, h_context) Tuple
    }

    class HRMAgent {
        +HRMConfig config
        +str device
        +Linear input_proj
        +ModuleList h_module
        +LModule l_module
        +AdaptiveComputationTime act
        +Sequential integrate
        +forward(x, max_steps, return_decomposition) HRMOutput
        +decompose_problem(query, state) List~SubProblem~
        +get_parameter_count() int
    }

    class HRMLoss {
        +float task_weight
        +float ponder_weight
        +float consistency_weight
        +forward(hrm_output, predictions, targets, task_loss_fn) Tuple
    }

    HRMAgent --> HRMConfig : configured by
    HRMAgent --> HRMOutput : produces
    HRMAgent --> HModule : contains
    HRMAgent --> LModule : contains
    HRMAgent --> AdaptiveComputationTime : uses
    HRMOutput --> SubProblem : contains
    HRMLoss --> HRMOutput : evaluates
```

### 4.2 TRM Agent Class Structure

```mermaid
classDiagram
    class TRMConfig {
        +int latent_dim
        +int num_recursions
        +int hidden_dim
        +bool deep_supervision
        +float supervision_weight_decay
        +float convergence_threshold
        +int min_recursions
    }

    class TRMOutput {
        +Tensor final_prediction
        +List~Tensor~ intermediate_predictions
        +int recursion_depth
        +bool converged
        +int convergence_step
        +List~float~ residual_norms
    }

    class RecursiveBlock {
        +TRMConfig config
        +Sequential transform
        +Parameter residual_scale
        +forward(x, iteration) Tensor
    }

    class DeepSupervisionHead {
        +int latent_dim
        +int output_dim
        +Sequential head
        +forward(x) Tensor
    }

    class TRMAgent {
        +TRMConfig config
        +str device
        +int output_dim
        +Sequential encoder
        +RecursiveBlock recursive_block
        +ModuleList supervision_heads
        +DeepSupervisionHead output_head
        +forward(x, num_recursions, check_convergence) TRMOutput
        +refine_solution(initial_prediction, num_recursions, convergence_threshold) Tuple
        +get_parameter_count() int
    }

    class TRMLoss {
        +Module task_loss_fn
        +float supervision_weight_decay
        +float final_weight
        +forward(trm_output, targets) Tuple
    }

    class TRMRefinementWrapper {
        +TRMAgent trm_agent
        +str device
        +refine(predictions, num_iterations, return_path) Tensor
        +get_refinement_stats(predictions) dict
    }

    TRMAgent --> TRMConfig : configured by
    TRMAgent --> TRMOutput : produces
    TRMAgent --> RecursiveBlock : contains
    TRMAgent --> DeepSupervisionHead : contains
    TRMLoss --> TRMOutput : evaluates
    TRMRefinementWrapper --> TRMAgent : wraps
```

### 4.3 Neural MCTS Class Structure

```mermaid
classDiagram
    class MCTSConfig {
        +int num_simulations
        +float c_puct
        +float dirichlet_epsilon
        +float dirichlet_alpha
        +int temperature_threshold
        +float temperature_init
        +float temperature_final
        +float virtual_loss
        +int num_parallel
    }

    class GameState {
        <<abstract>>
        +get_legal_actions() List
        +apply_action(action) GameState
        +is_terminal() bool
        +get_reward(player) float
        +to_tensor() Tensor
        +get_canonical_form(player) GameState
        +get_hash() str
    }

    class NeuralMCTSNode {
        +GameState state
        +NeuralMCTSNode parent
        +Any action
        +float prior
        +int visit_count
        +float value_sum
        +float virtual_loss
        +Dict children
        +bool is_expanded
        +bool is_terminal
        +float value
        +expand(policy_probs, valid_actions) void
        +select_child(c_puct) Tuple
        +add_virtual_loss(virtual_loss) void
        +revert_virtual_loss(virtual_loss) void
        +update(value) void
        +get_action_probs(temperature) Dict
    }

    class NeuralMCTS {
        +Module network
        +MCTSConfig config
        +str device
        +Dict cache
        +int cache_hits
        +int cache_misses
        +add_dirichlet_noise(policy_probs, epsilon, alpha) ndarray
        +evaluate_state(state, add_noise) Tuple
        +search(root_state, num_simulations, temperature, add_root_noise) Tuple
        -_simulate(node) float
        +select_action(action_probs, temperature, deterministic) Any
        +clear_cache() void
        +get_cache_stats() dict
    }

    class MCTSExample {
        +Tensor state
        +ndarray policy_target
        +float value_target
        +int player
    }

    class SelfPlayCollector {
        +NeuralMCTS mcts
        +MCTSConfig config
        +play_game(initial_state, temperature_threshold) List~MCTSExample~
        +generate_batch(num_games, initial_state_fn) List~MCTSExample~
    }

    NeuralMCTS --> MCTSConfig : configured by
    NeuralMCTS --> NeuralMCTSNode : manages
    NeuralMCTSNode --> GameState : represents
    SelfPlayCollector --> NeuralMCTS : uses
    SelfPlayCollector --> MCTSExample : produces
```

### 4.4 Policy-Value Network Class Structure

```mermaid
classDiagram
    class NeuralNetworkConfig {
        +int num_res_blocks
        +int num_channels
        +int policy_conv_channels
        +int policy_fc_dim
        +int value_conv_channels
        +int value_fc_hidden
        +bool use_batch_norm
        +float dropout
        +float weight_decay
        +int input_channels
        +int action_size
    }

    class ResidualBlock {
        +int channels
        +bool use_batch_norm
        +Conv2d conv1
        +BatchNorm2d bn1
        +Conv2d conv2
        +BatchNorm2d bn2
        +forward(x) Tensor
    }

    class PolicyHead {
        +int input_channels
        +int policy_conv_channels
        +int action_size
        +int board_size
        +Conv2d conv
        +BatchNorm2d bn
        +Linear fc
        +forward(x) Tensor
    }

    class ValueHead {
        +int input_channels
        +int value_conv_channels
        +int value_fc_hidden
        +int board_size
        +Conv2d conv
        +BatchNorm2d bn
        +Linear fc1
        +Linear fc2
        +forward(x) Tensor
    }

    class PolicyValueNetwork {
        +NeuralNetworkConfig config
        +int board_size
        +Conv2d conv_input
        +BatchNorm2d bn_input
        +ModuleList res_blocks
        +PolicyHead policy_head
        +ValueHead value_head
        +forward(x) Tuple
        +predict(state) Tuple
        +get_parameter_count() int
    }

    class AlphaZeroLoss {
        +float value_loss_weight
        +forward(policy_logits, value, target_policy, target_value) Tuple
    }

    class MLPPolicyValueNetwork {
        +int state_dim
        +int action_size
        +List~int~ hidden_dims
        +bool use_batch_norm
        +float dropout
        +Sequential shared_network
        +Sequential policy_head
        +Sequential value_head
        +forward(x) Tuple
        +get_parameter_count() int
    }

    PolicyValueNetwork --> NeuralNetworkConfig : configured by
    PolicyValueNetwork --> ResidualBlock : contains
    PolicyValueNetwork --> PolicyHead : contains
    PolicyValueNetwork --> ValueHead : contains
    AlphaZeroLoss --> PolicyValueNetwork : evaluates
    MLPPolicyValueNetwork --|> PolicyValueNetwork : alternative
```

### 4.5 Training Orchestrator Class Structure

```mermaid
classDiagram
    class SystemConfig {
        +HRMConfig hrm
        +TRMConfig trm
        +MCTSConfig mcts
        +NeuralNetworkConfig neural_net
        +TrainingConfig training
        +str device
        +int seed
        +bool use_mixed_precision
        +bool gradient_checkpointing
        +bool compile_model
        +bool distributed
        +to_dict() dict
        +from_dict(config_dict) SystemConfig
        +save(path) void
        +load(path) SystemConfig
    }

    class UnifiedTrainingOrchestrator {
        +SystemConfig config
        +Callable initial_state_fn
        +int board_size
        +str device
        +PerformanceMonitor monitor
        +HRMAgent hrm_agent
        +TRMAgent trm_agent
        +PolicyValueNetwork policy_value_net
        +NeuralMCTS mcts
        +SelfPlayCollector self_play_collector
        +Optimizer pv_optimizer
        +Optimizer hrm_optimizer
        +Optimizer trm_optimizer
        +Scheduler pv_scheduler
        +AlphaZeroLoss pv_loss_fn
        +HRMLoss hrm_loss_fn
        +TRMLoss trm_loss_fn
        +PrioritizedReplayBuffer replay_buffer
        +GradScaler scaler
        +int current_iteration
        +float best_win_rate
        -_initialize_components() void
        -_setup_optimizers() void
        -_setup_paths() void
        +train_iteration(iteration) Dict
        -_generate_self_play_data() List~Experience~
        -_train_policy_value_network() Dict
        -_train_hrm_agent() Dict
        -_train_trm_agent() Dict
        -_evaluate() Dict
        -_save_checkpoint(iteration, metrics, is_best) void
        -_log_metrics(iteration, metrics) void
        +train(num_iterations) void
        +load_checkpoint(path) void
    }

    class PerformanceMonitor {
        +int window_size
        +bool enable_gpu_monitoring
        +float alert_threshold_ms
        +deque metrics_history
        +Dict _metric_queues
        +int total_inferences
        +int slow_inference_count
        +log_timing(stage, elapsed_ms) void
        +log_memory() void
        +log_loss(policy_loss, value_loss, total_loss) void
        +log_mcts_stats(cache_hit_rate, simulations) void
        +log_inference(total_time_ms) void
        +get_stats(metric_name) Dict
        +get_current_memory() Dict
        +alert_if_slow() void
        +print_summary() void
        +export_to_wandb(step) Dict
        +reset() void
    }

    class PrioritizedReplayBuffer {
        +int capacity
        +float alpha
        +float beta_start
        +int beta_frames
        +int frame
        +List buffer
        +ndarray priorities
        +int position
        +int size
        -_get_beta() float
        +add(experience, priority) void
        +add_batch(experiences, priorities) void
        +sample(batch_size) Tuple
        +update_priorities(indices, priorities) void
    }

    class Experience {
        +Tensor state
        +ndarray policy
        +float value
        +dict metadata
    }

    UnifiedTrainingOrchestrator --> SystemConfig : configured by
    UnifiedTrainingOrchestrator --> PerformanceMonitor : uses
    UnifiedTrainingOrchestrator --> PrioritizedReplayBuffer : uses
    PrioritizedReplayBuffer --> Experience : stores
```

---

## Deployment Architecture

### Production Deployment Diagram

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Inference Namespace"
            subgraph "Inference Pods (3 replicas)"
                API1[FastAPI Server<br/>Pod 1<br/>4 CPU, 1 GPU, 16GB]
                API2[FastAPI Server<br/>Pod 2<br/>4 CPU, 1 GPU, 16GB]
                API3[FastAPI Server<br/>Pod 3<br/>4 CPU, 1 GPU, 16GB]
            end

            LB[Load Balancer<br/>Service]
            Ingress[Ingress Controller<br/>NGINX]
        end

        subgraph "Training Namespace"
            subgraph "Training Job"
                Train1[Training Pod 1<br/>8 CPU, 2 GPU, 32GB]
                Train2[Training Pod 2<br/>8 CPU, 2 GPU, 32GB]
                Train3[Training Pod 3<br/>8 CPU, 2 GPU, 32GB]
                Train4[Training Pod 4<br/>8 CPU, 2 GPU, 32GB]
            end
        end

        subgraph "Storage"
            PVC[Persistent Volume<br/>Checkpoints & Data<br/>500GB SSD]
        end
    end

    subgraph "External Services"
        S3[Cloud Storage<br/>S3/MinIO<br/>Long-term storage]
        WandB[Weights & Biases<br/>Experiment Tracking]
        Prometheus[Prometheus<br/>Metrics Collection]
        Grafana[Grafana<br/>Dashboards]
    end

    subgraph "Client Layer"
        WebApp[Web Application]
        MobileApp[Mobile App]
        CLI[CLI Tool]
    end

    WebApp -->|HTTPS| Ingress
    MobileApp -->|HTTPS| Ingress
    CLI -->|HTTPS| Ingress

    Ingress -->|Routes| LB
    LB -->|Distributes| API1
    LB -->|Distributes| API2
    LB -->|Distributes| API3

    API1 -->|Loads models| PVC
    API2 -->|Loads models| PVC
    API3 -->|Loads models| PVC

    Train1 -->|Saves to| PVC
    Train2 -->|Saves to| PVC
    Train3 -->|Saves to| PVC
    Train4 -->|Saves to| PVC

    Train1 -->|Archives to| S3
    Train1 -->|Logs to| WandB

    API1 -->|Metrics to| Prometheus
    Train1 -->|Metrics to| Prometheus
    Prometheus -->|Visualized in| Grafana

    style API1 fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
    style Train1 fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
```

---

## Data Flow Diagrams

### Training Data Flow

```mermaid
sequenceDiagram
    participant Orch as Training Orchestrator
    participant SelfPlay as Self-Play Collector
    participant MCTS as Neural MCTS
    participant PVNet as Policy-Value Net
    participant Buffer as Replay Buffer
    participant Trainer as Network Trainer
    participant Storage as Checkpoint Storage

    Note over Orch: Training Iteration Start

    Orch->>SelfPlay: Generate N games

    loop For each game
        SelfPlay->>MCTS: Run search from state
        MCTS->>PVNet: Evaluate state
        PVNet-->>MCTS: Policy + Value
        MCTS-->>SelfPlay: Action probabilities
        SelfPlay->>SelfPlay: Select action, update state
    end

    SelfPlay-->>Buffer: Store experiences

    Orch->>Buffer: Sample mini-batch
    Buffer-->>Trainer: (states, policies, values)

    Trainer->>PVNet: Forward pass
    PVNet-->>Trainer: Predictions
    Trainer->>Trainer: Compute loss
    Trainer->>PVNet: Backward + update

    Orch->>Orch: Evaluate model

    alt Model improved
        Orch->>Storage: Save checkpoint
    end

    Note over Orch: Iteration Complete
```

### Inference Data Flow

```mermaid
sequenceDiagram
    participant Client as Client App
    participant API as FastAPI Server
    participant Engine as Inference Engine
    participant HRM as HRM Agent
    participant MCTS as Neural MCTS
    participant TRM as TRM Agent
    participant PVNet as Policy-Value Net
    participant Cache as Eval Cache

    Client->>API: POST /inference<br/>{state, options}
    API->>API: Validate request
    API->>Engine: Process request

    alt HRM decomposition requested
        Engine->>HRM: Decompose problem
        HRM-->>Engine: Subproblems
    end

    alt MCTS search requested
        Engine->>MCTS: Run search

        loop MCTS simulations
            MCTS->>Cache: Check cache
            alt Cache miss
                MCTS->>PVNet: Evaluate state
                PVNet-->>MCTS: Policy + Value
                MCTS->>Cache: Store result
            else Cache hit
                Cache-->>MCTS: Cached result
            end
        end

        MCTS-->>Engine: Action probabilities + Value
    end

    alt TRM refinement requested
        Engine->>TRM: Refine solution
        TRM-->>Engine: Refined output
    end

    Engine->>Engine: Aggregate results
    Engine->>Engine: Measure performance
    Engine-->>API: Results + stats
    API-->>Client: JSON response
```

### Self-Play Game Flow

```mermaid
flowchart TD
    Start([Game Start]) --> InitState[Initialize Game State]

    InitState --> CheckTerm{Terminal?}

    CheckTerm -->|No| RunMCTS[Run MCTS<br/>num_simulations]
    CheckTerm -->|Yes| CalcOutcome[Calculate Outcome]

    RunMCTS --> AddNoise{First 30 moves?}
    AddNoise -->|Yes| HighTemp[Temperature = 1.0<br/>Exploration]
    AddNoise -->|No| LowTemp[Temperature = 0.1<br/>Greedy]

    HighTemp --> SelectAction[Select Action<br/>from MCTS policy]
    LowTemp --> SelectAction

    SelectAction --> StoreExample[Store Training Example<br/>state, policy, player]

    StoreExample --> ApplyAction[Apply Action<br/>to State]
    ApplyAction --> SwitchPlayer[Switch Player]
    SwitchPlayer --> CheckTerm

    CalcOutcome --> AssignValues[Assign Outcome<br/>to All Positions]
    AssignValues --> ReturnExamples[Return Examples]
    ReturnExamples --> End([Game End])

    style Start fill:#95E1D3,stroke:#38A89D
    style End fill:#95E1D3,stroke:#38A89D
    style RunMCTS fill:#3498DB,stroke:#1F618D,color:#fff
    style StoreExample fill:#F39C12,stroke:#B9770E
```

---

## Summary

This C4 architecture documentation provides:

1. **Context Diagram**: System boundaries and external interactions
2. **Container Diagram**: High-level technical components
3. **Component Diagrams**: Detailed internal structure of each container
4. **Code Diagrams**: Class structures and relationships
5. **Deployment Architecture**: Production deployment on Kubernetes
6. **Data Flow Diagrams**: Request/response and training flows

### Key Architectural Patterns

- **Microservices**: Separate training and inference services
- **Event-Driven**: Async training pipeline with callbacks
- **Repository**: Centralized checkpoint and config management
- **Strategy**: Pluggable MCTS policies and loss functions
- **Observer**: Performance monitoring and experiment tracking
- **Factory**: Component creation with configuration
- **Singleton**: Shared cache and monitoring instances

### Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **Core ML** | PyTorch 2.1+, NumPy, SciPy |
| **Orchestration** | Python AsyncIO, LangGraph |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Storage** | S3/MinIO, Persistent Volumes |
| **Monitoring** | Weights & Biases, Prometheus, Grafana |
| **Testing** | pytest, pytest-asyncio |
| **Deployment** | Kubernetes, Docker |
| **Code Quality** | Black, Ruff, MyPy, pre-commit |
