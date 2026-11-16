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

**Limits.** One dataset, whose 41 kinds of question are the same in every contract. Mean-centring uses the passages
of the set being searched (no labels). The yes/no step is too slow as run. Not yet measured: whether better search
raises the answer score end to end.
