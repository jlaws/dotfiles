# Paper Classification Decision Tree

## Relevance Gate (apply before routing)

The library is **selective**. Only papers whose *primary contribution* is a general,
reusable AI/ML advance in a focus area are kept. Judge every paper by **what it
contributes, not which folder it could fit** — then assign one `relevance_class`:

| `relevance_class` | Meaning | Decision |
|---|---|---|
| `core` | General, reusable AI/ML advance in a focus area (agentic AI, LLMs, neural architectures, reasoning, RL, multiagent, alignment, efficient computation, scaling, memory/retrieval, interpretability, representation learning, optimization theory, neuro-symbolic, data curation, continual/transfer). | **KEEP** (if quality passes) |
| `applied` | Domain-specific application — the ML method is a tool applied to medicine, biology, genomics, chemistry, finance, law, climate, education, etc. The contribution is the domain result, not the method. | **DELETE** |
| `peripheral` | Not advancing LLMs/agents/core architectures: pure robotics control, speech/TTS/ASR, image/video/3D **generation as the output**, standalone computer vision. | **DELETE** |
| `off-topic` | HCI/user studies, social-science or labor/impact analyses, opinion/position essays, tutorials, proceedings/prefaces, or unrelated work. | **DELETE** |

**Only `core` papers are kept.** DELETE `applied`, `peripheral`, and `off-topic`.

Two clarifying rules:
- **Generative models — keep methods, reject outputs.** A new sampler, flow-matching
  theory, or training method that generalizes is `core`. A paper whose contribution is
  generating specific images/video/3D is `peripheral`.
- **Domain as evaluation vehicle.** A paper is still `core` if its methodological
  contribution generalizes to the focus areas and the domain is merely where it is
  evaluated. It is `applied` when the contribution is the domain-specific result.

When genuinely uncertain after applying these rules, lean toward **DELETE** — the
library is curated.

## Routing (topic folder — applies once a paper is KEPT)

Apply top-to-bottom, **first match wins**.

## Priority Rules

| # | Question | Destination |
|---|----------|-------------|
| 1a | Multi-agent coordination/communication/emergence as core contribution? | `multiagent-systems/` |
| 1b | Agent systems (tool use, agent architectures, benchmarks, agent memory)? | `agentic-ai/` |
| 2 | Safety, alignment, adversarial attacks, jailbreaking, ethics, governance? | `alignment-and-safety/` |
| 3 | Reasoning methods (CoT, ToT, math, verification, test-time compute for reasoning)? | `reasoning-and-planning/` |
| 4 | RL methods/theory (RLHF methodology, DPO, reward models, policy optimization)? | `reinforcement-learning/` |
| 5 | Retrieval, RAG, external memory architectures, knowledge graphs for retrieval? | `memory-and-retrieval/` |
| 6 | Efficient inference/compression (speculative decoding, quantization, pruning, distillation)? | `efficient-computation/` |
| 7 | Novel neural architecture (transformer variants, SSMs, linear attention, MoE architecture)? | `neural-architectures/` |
| 8 | LLM training, evaluation, capabilities, prompting, instruction tuning? | `large-language-models/` |
| 9 | Scaling laws, emergent capabilities, compute-optimal training? | `scaling-and-emergent-capabilities/` |
| 10 | Data curation, selection, synthetic data generation? | `data-curation-and-synthetic-data/` |
| 11 | Optimization theory (optimizers, loss landscapes, training dynamics)? | `optimization-theory/` |
| 12 | Interpretability (mechanistic interp, probing, grokking, circuits)? | `interpretability/` |
| 13 | Representation learning (embeddings, SSL, world models, contrastive)? | `representation-learning/` |
| 14 | Neuro-symbolic (hybrid neural-symbolic, program synthesis, theorem proving)? | `neuro-symbolic-ai/` |
| 15 | Biologically-inspired (evolutionary, brain-inspired, Hebbian)? | `biologically-inspired-ai/` |
| 16 | Continual/transfer learning (forgetting, domain adaptation, lifelong)? | `continual-and-transfer-learning/` |
| 17 | Non-LLM generative models (diffusion, VAEs, flows)? | `probabilistic-and-generative-models/` |
| 18 | ML infrastructure/systems (distributed training, serving, hardware)? | `systems-for-ml/` |
| 19 | Domain: vision → `computer-vision/`, robotics → `robotics/`, SE/code → `software-engineering/`, HCI → `human-computer-interaction/`, audio → `sound/`, info theory → `information-theory/` |

## Key Tiebreakers

| Ambiguous case | Goes to | Rationale |
|---|---|---|
| Agentic RL (Agent-R1, RAGEN) | `agentic-ai/` | Agent is the contribution, RL is the method |
| RLHF/DPO methodology papers | `reinforcement-learning/` | RL method is the contribution |
| Constitutional AI, safety RLHF | `alignment-and-safety/` | Safety framing dominates |
| RAG papers | `memory-and-retrieval/` | Not classic IR |
| Scaling laws for LLMs | `scaling-and-emergent-capabilities/` | More specific than LLMs |
| Multi-agent RL (MARL) | `multiagent-systems/` | Multi-agent is the focus |
| Agent memory papers | `agentic-ai/` | Agent capability |
| Process reward models | `reasoning-and-planning/` | Reasoning verification method |
| Reward model surveys | `reinforcement-learning/` | Core RL concept |
| Coding agents (SWE-bench) | `agentic-ai/` | Agent research; SE methodology stays in SE |
| Medical/domain LLMs | `large-language-models/` | LLM is the research contribution |
| Benchmark papers | By what they evaluate | Agent benchmarks → agentic-ai, etc. |
| VLM papers (BLIP-2, LLaVA) | `large-language-models/` | LLM with vision modality |
