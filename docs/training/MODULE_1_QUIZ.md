# Module 1: Architecture Quiz

**Time Limit:** 15 minutes
**Passing Score:** 80% (16/20 correct)
**Format:** Multiple choice and short answer

---

## Section A: System Context (Questions 1-5)

### Question 1
What is the primary purpose of the LangGraph Multi-Agent MCTS framework?

A) Build chatbots for customer service
B) Provide tactical and cybersecurity analysis using multi-agent reasoning
C) Train language models from scratch
D) Replace human analysts entirely

**Answer:** ____

---

### Question 2
Which of the following is an **external** dependency (outside the system boundary)?

A) HRM Agent
B) LangGraph Orchestration Layer
C) OpenAI API
D) Vector Storage

**Answer:** ____

---

### Question 3
What authentication mechanism does the API Gateway use?

A) Basic Auth
B) OAuth 2.0
C) JWT (JSON Web Tokens)
D) API Keys only

**Answer:** ____

---

### Question 4
Which component is responsible for rate limiting and request validation?

A) HRM Agent
B) API Gateway
C) LangGraph Orchestration Layer
D) Storage Layer

**Answer:** ____

---

### Question 5
True or False: The system can horizontally scale because agents are stateless.

**Answer:** ____

---

## Section B: Container Architecture (Questions 6-10)

### Question 6
Which file contains the main FastAPI routes for the inference server?

A) src/agents/hrm_agent.py
B) src/api/inference_server.py
C) examples/langgraph_multi_agent_mcts.py
D) src/storage/pinecone_integration.py

**Answer:** ____

---

### Question 7
What data structure does LangGraph use to maintain workflow state?

A) JSON string
B) TypedDict (AgentState)
C) Pandas DataFrame
D) Redis hash

**Answer:** ____

---

### Question 8
Which storage backends are supported for vector embeddings? (Select all that apply)

A) Pinecone
B) FAISS
C) Elasticsearch
D) MongoDB

**Answer:** ____

---

### Question 9
In the container architecture, which layer sits between the API Gateway and the individual agents?

A) Storage Layer
B) LangGraph Orchestration Layer
C) Observability Layer
D) Cache Layer

**Answer:** ____

---

### Question 10
What happens if the vector database is unavailable?

A) The entire system crashes
B) Queries fall back to in-memory search
C) Error is returned to the user with retry logic
D) HRM Agent generates responses without context

**Answer:** ____

---

## Section C: Agent Responsibilities (Questions 11-15)

### Question 11
Match each agent to its primary responsibility:

1. HRM Agent
2. TRM Agent
3. MCTS Engine

A) Iterative refinement through self-critique
B) Task decomposition and objective identification
C) Tree search and win probability estimation

**Answers:**
- HRM → ____
- TRM → ____
- MCTS → ____

---

### Question 12
What is the output of the HRM Agent?

A) A refined final answer
B) A decision tree with win probabilities
C) A structured task list with objectives
D) A vector embedding

**Answer:** ____

---

### Question 13
How does TRM determine when to stop refining?

A) After exactly 5 iterations
B) When the user manually stops it
C) When improvements plateau (convergence detection)
D) When confidence score reaches 1.0

**Answer:** ____

---

### Question 14
In MCTS, what does the UCB1 algorithm balance?

A) Speed and accuracy
B) Exploitation and exploration
C) Cost and performance
D) Latency and throughput

**Answer:** ____

---

### Question 15
True or False: MCTS is always required for every query, regardless of complexity.

**Answer:** ____

---

## Section D: LangGraph Integration (Questions 16-20)

### Question 16
What are the main components of a LangGraph state machine? (Select all that apply)

A) Nodes (functions)
B) Edges (transitions)
C) State schema (TypedDict)
D) Database tables

**Answer:** ____

---

### Question 17
What is the purpose of "conditional edges" in LangGraph?

A) To speed up execution
B) To enable dynamic routing based on state
C) To handle errors
D) To save state to disk

**Answer:** ____

---

### Question 18
In the AgentState TypedDict, which field tracks the current execution phase?

A) status
B) current_phase
C) step
D) agent_name

**Answer:** ____

---

### Question 19
Why was LangGraph chosen over CrewAI for this framework?

A) LangGraph is cheaper
B) LangGraph provides explicit state control and better tracing
C) CrewAI doesn't support LLMs
D) CrewAI is deprecated

**Answer:** ____

---

### Question 20
Short Answer (2-3 sentences): Describe the complete flow of a query from the API Gateway to the final response.

**Answer:**
_________________________________________________________________________
_________________________________________________________________________
_________________________________________________________________________

---

## Answer Key (For Instructors Only)

### Section A: System Context
1. **B** - Tactical and cybersecurity analysis
2. **C** - OpenAI API (external LLM provider)
3. **C** - JWT
4. **B** - API Gateway
5. **True**

### Section B: Container Architecture
6. **B** - src/api/inference_server.py
7. **B** - TypedDict (AgentState)
8. **A, B** - Pinecone and FAISS
9. **B** - LangGraph Orchestration Layer
10. **C** - Error is returned with retry logic

### Section C: Agent Responsibilities
11. **HRM → B, TRM → A, MCTS → C**
12. **C** - Structured task list with objectives
13. **C** - Convergence detection
14. **B** - Exploitation and exploration
15. **False** - MCTS is conditionally invoked

### Section D: LangGraph Integration
16. **A, B, C** - Nodes, Edges, State schema
17. **B** - Dynamic routing based on state
18. **B** - current_phase
19. **B** - Explicit state control and better tracing
20. **Sample Answer:** "A query enters through the API Gateway, which validates and routes it to the LangGraph Orchestration Layer. The LangGraph state machine invokes HRM for task decomposition, then TRM for refinement, and optionally MCTS for decision tree search. The final result is assembled from agent outputs and returned to the user via the API."

---

## Scoring Guide

- **Questions 1-19:** 1 point each
- **Question 20:** 1 point (must include API Gateway → LangGraph → HRM → TRM → (optional MCTS) → Response)

**Total:** 20 points
**Passing:** 16+ points (80%)

---

## After the Quiz

### If you passed (80%+):
Congratulations! You're ready for Module 2. Continue to [MODULE_2_AGENTS.md](MODULE_2_AGENTS.md).

### If you scored 70-79%:
Review the sections where you missed questions:
- Section A: Re-read [architecture.md](../architecture.md)
- Section B: Review [C4_ARCHITECTURE.md](../C4_ARCHITECTURE.md)
- Section C: Review [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md)
- Section D: Review [langgraph_mcts_architecture.md](../langgraph_mcts_architecture.md)

Retake the quiz after review.

### If you scored below 70%:
Schedule office hours with an instructor to review architecture concepts before retaking the quiz.
