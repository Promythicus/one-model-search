# 05 Search for contracts with frozen models, ending at one model

**Question.** Can clause search be sharpened without training, and can the model that answers (the 27B file) also do
the search, so that nothing is swapped on the GPU?

**Data and score.** CUAD, split by contract (seed 3): 40 practice, 62 held out (741 questions, 6,724 passages).
Score: share of the marked clauses that fall inside the 12 passages read (about 6,000 characters). Same passages and
scoring as research line A.

**Results, held out** (`out/head.jsonl`, `out/fit.json`, `out/types.json`, `out/own27_pick.json`, `out/rr27.json`).

| Search | Fitted on contracts? | Clauses found |
| ------ | -------------------- | ------------- |
| Qwen3-8B raw states (`clm.py`) | no | 19.0% |
| Qwen3-8B + released CLM-v0.1-8B head | no | 29.0% |
| 0.6B embedder | no | 67.4% |
| 0.6B embedder + stock reranker | no | 74.3% |
| reranker + closed-form direction at its layer 15 (`rr.py`) | yes | 79.7% |
| embedder + closed-form bilinear add-on (`head.py linear`) | yes | 87.7% |
| embedder + trained contrastive add-on (`head.py train`) | yes | 87.6% |
| **the 27B's own states, layer 36, mean over the passage's tokens (`own27.py`)** | no | **75.8%** |
| **the same, then the 27B's own yes/no on the 30 best (`rr27.py`)** | no | **87.8%** |

- The fitted add-ons do not carry to new questions (`head.py types`): fitted on half the kinds of question, they
  raise the kinds they saw from 65.6% to 90.0% and the kinds they never saw from 70.8% to 73.5%. Dropped as a route.
- The 27B's own states: the layer (36 of 64) was chosen on the practice contracts (73.0% there). Layers 36-40 and 52
  are within two points of each other on both sets; the last layer and the "in one word" last-token form are poor
  (24-37%).
- The released general CLM head is weak here. Caveats: 8-bit load, and its prompt format was not confirmed.

**Cost of the one-model route.** States: about 16 passages a second, once per contract set. Yes/no: 0.23 s a passage,
6.8 s a question at 30 passages (the outline method of A-02: 2.7 s a question).

**End to end** (`e2e27.py`, `out/e2e/scores.json`; the ten pilot contracts of line A, 410 questions, 248 marked clauses).

| Way of reading | Right passage read | Clauses quoted | Seconds a question |
| -------------- | ------------------ | -------------- | ------------------ |
| own states, 12 passages | 86.2% | 73.0% | 3.0 |
| own states then own yes/no, 12 passages | 95.1% | 82.7% | 3.4 + 6.1 |
| own states then own yes/no, 24 passages | 95.1% | 70.2% | 3.9 + 6.1 |

Line A on the same contracts: best separate search 70.6%, outline 80.6-81.9%, whole contract 82.3%. Reading 24
scattered passages is worse than reading 12.

**Speed of the yes/no step** (`out/rr27_speed.jsonl`, `out/rr27-short-every3.json`). 30 passages sent together: 8.6 s
a question with 1, 4 or 8 server slots; 5.9 s with the server's RAM prompt cache off; 5.6 s with 8 slots. A shorter
prompt (no system prompt, the clause's name only): 5.5 s against 6.8 s, at 85.7% against 87.5% on the same 21
contracts. Fewer passages: 20 give 84.8%, 15 give 79.4%. Not solved: the limit is reading about 220 tokens a passage.

**Limits.** One dataset, whose 41 kinds of question are the same in every contract. Mean-centring uses the passages
of the set being searched (no labels). The yes/no step is too slow as run. Not yet measured: whether better search
raises the answer score end to end.
