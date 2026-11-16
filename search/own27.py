#!/usr/bin/env python3
"""One model for everything: can the 27B file we answer with also do the search, from its own internal states?
Nothing is trained or fitted. (Experiment A-02's outline routing is the other one-model route: 81% of clauses.)
    python own27.py prompts    passages of the 62 held-out contracts and the questions, in two forms -> out/own27/<form>/in.jsonl
        mean   the bare text; the state averaged over its tokens
        eol    the text followed by "... in one word:"; the state at the last token (a known trick for getting a
               summary state out of a language model without training)
    (then tools/probe on each form, see run5.sh)
    python own27.py score      per form and layer: cosine after removing the passages' common part, scored like head.py -> out/own27.json
Every layer is reported; none is chosen there. Choosing the layer fairly (run6.sh):
    python own27.py prompts2   mean form, the 40 practice and the 62 held-out contracts -> out/own27/pick/<set>/in.jsonl
    python own27.py pick       layers PICK scored on both; the layer is the best on practice -> out/own27_pick.json"""
import json, os, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "..", "cuad-retrieval"); sys.path.insert(0, E1)
import retr, head
OUT = os.path.join(H, "out", "own27"); LAYERS = (24, 32, 40, 48, 56, 63); MARK = "␞"
FORMS = {"mean": (lambda t: MARK + t, lambda q: MARK + retr.short(q)),
         "eol": (lambda t: f"Passage from a contract:\n{t}\n\nThe subject of this passage in one word:", lambda q: f"Question about a contract:\n{retr.short(q)}\n\nThe subject of this question in one word:")}


def held():
    return [c for c in head.setup() if c["set"] == "held"]


def prompts():
    C = held(); qs = sorted({it["text"] for c in C for it in c["items"]})
    for f, (fp, fq) in FORMS.items():
        os.makedirs(os.path.join(OUT, f), exist_ok=True)
        with open(os.path.join(OUT, f, "in.jsonl"), "w") as fh:
            for c in C:
                for t in c["texts"]:
                    fh.write(json.dumps({"text": fp(t)}) + "\n")
            for q in qs:
                fh.write(json.dumps({"text": fq(q)}) + "\n")
    print(sum(len(c["texts"]) for c in C), "passages", len(qs), "questions")


def score():
    C = held(); qs = sorted({it["text"] for c in C for it in c["items"]}); qi = {q: i for i, q in enumerate(qs)}; n = sum(len(c["texts"]) for c in C); rep = {}
    for f in FORMS:
        m = json.load(open(os.path.join(OUT, f, "meta.json"))); L, d = m["shape"]["l_out"]; X = np.memmap(os.path.join(OUT, f, "l_out.f32"), dtype=np.float32, mode="r").reshape(-1, L, d)
        for li, layer in enumerate(LAYERS):
            P = torch.tensor(np.array(X[:n, li])); Q = torch.tensor(np.array(X[n:, li])); mu = P.mean(0); nz = torch.nn.functional.normalize
            P, Q = nz(P - mu, dim=-1), nz(Q - Q.mean(0), dim=-1); at = 0; PC = {}
            for c in C:
                PC[c["ci"]] = P[at:at + len(c["texts"])]; at += len(c["texts"])
            rep[f"{f} L{layer}"] = head.measure(C, "held", lambda c, it: PC[c["ci"]] @ Q[qi[it["text"]]]); print(f, layer, rep[f"{f} L{layer}"], flush=True)
    json.dump(rep, open(os.path.join(H, "out", "own27.json"), "w"), indent=1)


PICK = (24, 28, 32, 36, 38, 40, 42, 44, 48, 52, 56, 60)


def sets():
    C = head.setup(); return {"practice": [c for c in C if c["set"] != "held"], "held": [c for c in C if c["set"] == "held"]}


def prompts2():
    fp, fq = FORMS["mean"]
    for name, C in sets().items():
        qs = sorted({it["text"] for c in C for it in c["items"]}); os.makedirs(os.path.join(OUT, "pick", name), exist_ok=True)
        with open(os.path.join(OUT, "pick", name, "in.jsonl"), "w") as fh:
            for c in C:
                for t in c["texts"]:
                    fh.write(json.dumps({"text": fp(t)}) + "\n")
            for q in qs:
                fh.write(json.dumps({"text": fq(q)}) + "\n")
        print(name, len(C), "contracts", sum(len(c["texts"]) for c in C), "passages", len(qs), "questions")


def pick():
    rep = {}; nz = torch.nn.functional.normalize
    for name, C in sets().items():
        qs = sorted({it["text"] for c in C for it in c["items"]}); qi = {q: i for i, q in enumerate(qs)}; n = sum(len(c["texts"]) for c in C)
        m = json.load(open(os.path.join(OUT, "pick", name, "meta.json"))); L, d = m["shape"]["l_out"]; X = np.memmap(os.path.join(OUT, "pick", name, "l_out.f32"), dtype=np.float32, mode="r").reshape(-1, L, d)
        for c in C:
            c["set"] = name
        rep[name] = {}
        for li, layer in enumerate(PICK):
            P = torch.tensor(np.array(X[:n, li])); Q = torch.tensor(np.array(X[n:, li])); P, Q = nz(P - P.mean(0), dim=-1), nz(Q - Q.mean(0), dim=-1); at = 0; PC = {}
            for c in C:
                PC[c["ci"]] = P[at:at + len(c["texts"])]; at += len(c["texts"])
            rep[name][layer] = head.measure(C, name, lambda c, it: PC[c["ci"]] @ Q[qi[it["text"]]]); print(name, layer, rep[name][layer], flush=True)
    best = max(PICK, key=lambda l: rep["practice"][l]["span6k"]); rep["layer_chosen_on_practice"] = best; rep["held_out_at_that_layer"] = rep["held"][best]
    json.dump(rep, open(os.path.join(H, "out", "own27_pick.json"), "w"), indent=1); print("chosen", best, rep["held"][best])


if __name__ == "__main__":
    {"prompts": prompts, "score": score, "prompts2": prompts2, "pick": pick}[sys.argv[1]]()
