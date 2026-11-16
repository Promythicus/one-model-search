#!/usr/bin/env python3
"""Sharpening the reranker (Qwen3-Reranker-0.6B) for contracts without training.
    python rr.py feats     for every CUAD question with an answer: its first 30 passages (experiment 01's best search),
                           the reranker's own score and its internal state at the last position, every layer -> out/feats.pt
    python rr.py fit       on the practice contracts only: per layer, the closed-form direction that best separates
                           passages that hold an answer from those that do not; scored on the held-out contracts -> out/fit.json
Split: the 102 test contracts shuffled with seed 3; first 40 practice, other 62 held out.
Measures as in experiment 01: hit = a passage holding part of an answer is among those read; span = share of marked
clauses in a passage read; 6k = 12 passages of 500 characters, 12k = 24."""
import json, os, random, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "..", "cuad-retrieval"); sys.path.insert(0, E1)
import bench, retr
OUT = os.path.join(H, "out"); FIRST, SPEC = 30, "500"
NAME = "Qwen/Qwen3-Reranker-0.6B"
PRE = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def split():
    idx = list(range(102)); random.Random(3).shuffle(idx)
    return set(idx[:40]), set(idx[40:])


def feats():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Dt = retr.data(); V, Q, qi = retr.vectors("qwen06", SPEC, Dt); retr._loaded.clear(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(NAME, padding_side="left"); m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    X, base, lab, grp, cid, spans = [], [], [], [], [], []; at = 0; g = 0
    for ci, (title, text, qs) in enumerate(Dt):
        ch, texts = retr.passages(title, text, SPEC)
        for q, gold in qs:
            first = np.argsort(-(V[at:at + len(ch)] @ Q["short"][qi[q]].astype(np.float32))).tolist()[:FIRST]
            for b in range(0, len(first), 15):
                t = [f"{PRE}<Instruct>: Given a question about a contract, retrieve the passages of the contract that answer it\n<Query>: {retr.short(q)}\n<Document>: {texts[i][:5000]}{SUF}" for i in first[b:b + 15]]
                x = tok(t, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    o = m(**x, output_hidden_states=True)
                X.append(torch.stack([h[:, -1] for h in o.hidden_states[1:]], 1).cpu()); lg = o.logits[:, -1].float(); base += (lg[:, yes] - lg[:, no]).tolist()
            for i in first:
                held = [ch[i][0] < ge and gs < ch[i][1] for gs, ge in gold]
                lab.append(any(held)); grp.append(g); cid.append(ci); spans.append(held)
            g += 1
        at += len(ch)
        if ci % 10 == 0:
            print("contract", ci, flush=True)
    torch.save({"X": torch.cat(X), "base": torch.tensor(base), "label": torch.tensor(lab), "group": torch.tensor(grp), "contract": torch.tensor(cid), "spans": spans,
                "readout": (m.get_output_embeddings().weight[yes] - m.get_output_embeddings().weight[no]).float().cpu()}, os.path.join(OUT, "feats.pt"))
    print("pairs", len(lab), "questions", g, "positives", sum(lab))


def rank_scores(score, F, keep):
    """hit and span at 12 and 24 passages, over the questions whose contract is in `keep`."""
    out = {"hit6k": 0, "span6k": 0, "hit12k": 0, "span12k": 0}; n = ns = 0
    for g in F["group"].unique().tolist():
        ix = (F["group"] == g).nonzero().flatten()
        if int(F["contract"][ix[0]]) not in keep:
            continue
        order = ix[score[ix].argsort(descending=True)].tolist(); nsp = len(F["spans"][ix[0]]); n += 1; ns += nsp
        for name, top in (("6k", 12), ("12k", 24)):
            got = [any(F["spans"][i][s] for i in order[:top]) for s in range(nsp)]
            out["hit" + name] += any(got); out["span" + name] += sum(got)
    return {k: round(v / (n if k.startswith("hit") else ns), 4) for k, v in out.items()} | {"questions": n}


def fit():
    F = torch.load(os.path.join(OUT, "feats.pt")); practice, held = split()
    tr = torch.tensor([int(c) in practice for c in F["contract"]]); X = F["X"].float(); y = F["label"]
    rep = {"pairs": len(y), "practice_pairs": int(tr.sum()), "base": {"practice": rank_scores(F["base"], F, practice), "held_out": rank_scores(F["base"], F, held)},
           "first_30_order": {"held_out": rank_scores(-torch.arange(len(y)).float(), F, held)}, "layers": []}
    for l in range(X.shape[1]):
        A = X[tr, l].double(); mu = A.mean(0); C = (A - mu).T @ (A - mu) / len(A)
        w = torch.linalg.solve(C + 0.1 * C.diag().mean() * torch.eye(C.shape[0], dtype=C.dtype), A[y[tr]].mean(0) - A[~y[tr]].mean(0)).float()
        s = X[:, l] @ w
        rep["layers"].append({"layer": l, "practice": rank_scores(s, F, practice), "held_out": rank_scores(s, F, held)})
        print(l, rep["layers"][-1]["held_out"], flush=True)
    json.dump(rep, open(os.path.join(OUT, "fit.json"), "w"), indent=1); print("base", rep["base"])


if __name__ == "__main__":
    {"feats": feats, "fit": fit}[sys.argv[1]]()
