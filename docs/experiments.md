# Experiments

Planned experiment families are: LLM-only, dense RAG, dense+sparse hybrid RAG, fixed vector+graph RAG, and full KG-CRAG. Ablations remove graph retrieval, sparse retrieval, adaptive routing, corrective retrieval, answer critique, or web fallback; RRF and weighted fusion are compared separately.

Every run must record the dataset split, random seed, configuration hash, prompt/model/index versions, latency, tool calls, and token or inference cost. Generated result files belong under `data/evaluation/results/` and are not committed by default.

