#!/usr/bin/env python3
"""End to end with one model: does the 27B's own search (its states, then its yes/no) raise the answer score?
Same ten pilot contracts, 410 questions, prompt and scoring as research line A (cuad-retrieval/bench.py,
outline-routing/e2e.py), so the rows can be set beside theirs. Nothing is fitted; layer 36 was chosen on the
practice contracts (own27.py).
    python e2e27.py prompts          500-character passages and the questions -> out/e2e/in.jsonl   (then tools/probe, layer 36)
    python e2e27.py orders           first 30 by own states, re-ordered by own yes/no (server up)   -> out/e2e/orders.json
    python e2e27.py run own12|own24|states12|states24      ask the model (server up)               -> out/e2e/<mode>.jsonl
    python e2e27.py score            -> out/e2e/scores.json"""
import json, os, sys, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "..", "cuad-retrieval"); sys.path.insert(0, E1)
import bench, retr
OUT = os.path.join(H, "out", "e2e"); MARK = "␞"; FIRST = 30; MODES = ("states12", "states24", "own12", "own24")


def cs():
    bench.CHUNK = 500
    return [(c["paragraphs"][0]["context"], [q["question"] for q in c["paragraphs"][0]["qas"]]) for c in bench.contracts()]


def layout():
    C = cs(); ch = [bench.chunks(t) for t, _ in C]; qs = sorted({q for _, Q in C for q in Q}); return C, ch, qs


def prompts():
    C, ch, qs = layout(); os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "in.jsonl"), "w") as f:
        for (t, _), c in zip(C, ch):
            for a, b in c:
                f.write(json.dumps({"text": MARK + t[a:b]}) + "\n")
        for q in qs:
            f.write(json.dumps({"text": MARK + retr.short(q)}) + "\n")
    print(sum(map(len, ch)), "passages", len(qs), "questions")


def orders():
    import rr27
    C, ch, qs = layout(); n = sum(map(len, ch)); m = json.load(open(os.path.join(OUT, "meta.json"))); L, d = m["shape"]["l_out"]
    X = np.fromfile(os.path.join(OUT, "l_out.f32"), dtype=np.float32).reshape(-1, L, d)[:, 0]; P, Q = X[:n], X[n:]
    P = P - P.mean(0); Q = Q - Q.mean(0); P /= np.linalg.norm(P, axis=1, keepdims=True); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    out = {"states": {}, "own": {}}; at = 0; secs = []
    for k, (text, Qs) in enumerate(C):
        p = P[at:at + len(ch[k])]; at += len(ch[k])
        for q in Qs:
            first = np.argsort(-(p @ Q[qs.index(q)])).tolist()[:FIRST]; t = time.time()
            s = [rr27.yes_no(rr27.prompt(text[a:b], q)) for a, b in (ch[k][j] for j in first)]; secs.append(time.time() - t)
            out["states"][f"{k}|{q}"] = [ch[k][j] for j in first[:24]]
            out["own"][f"{k}|{q}"] = [ch[k][first[j]] for j in sorted(range(len(first)), key=lambda j: -s[j])[:24]]
        print(k, round(sum(secs) / len(secs), 2), "s a question", flush=True)
    out["yes_no_seconds_a_question"] = round(sum(secs) / len(secs), 2); json.dump(out, open(os.path.join(OUT, "orders.json"), "w"))


def run(mode):
    O = json.load(open(os.path.join(OUT, "orders.json")))[mode.rstrip("0123456789")]; top = int(mode.lstrip("ownstae")); path = os.path.join(OUT, mode + ".jsonl")
    done = {(json.loads(l)["k"], json.loads(l)["question"]) for l in open(path)} if os.path.exists(path) else set(); f = open(path, "a")
    for k, (text, Q) in enumerate(cs()):
        for q in Q:
            if (k, q) in done:
                continue
            got = sorted(map(tuple, O[f"{k}|{q}"][:top]))
            a = bench.ask("EXCERPTS FROM THE CONTRACT:\n\n" + "\n\n---\n\n".join(text[x:y] for x, y in got), q)
            a.update(k=k, question=q, got=got); f.write(json.dumps(a) + "\n"); f.flush()
        print(mode, k, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "score":
        bench.OUT = OUT; json.dump(bench.score(MODES), open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    else:
        {"prompts": prompts, "orders": orders}.get(a[0], lambda: run(a[1]))()
