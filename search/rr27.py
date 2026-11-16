#!/usr/bin/env python3
"""One model for everything, second stage: the 27B as its own reranker. Nothing is trained or fitted.
First stage: the 27B's own states (own27.py), layer chosen on practice (out/own27_pick.json). The 30 best passages per
question are then put to the same file as a yes/no question; score = log P(yes) - log P(no) at the first answer token.
The passage comes first in the prompt and the question last, and calls are ordered by passage, so the server reuses
the passage it has already read and only the question is new work.
    python rr27.py ask      -> out/rr27_scores.json   (needs the server on :8082; can be re-run, keeps what it has)
    python rr27.py score    -> out/rr27.json          (held-out contracts; first stage alone, and re-ordered)
    python rr27.py speed <tag>   one question at a time, its 30 passages sent together (as a product would); 60 questions
                                 spread over the held-out set; checks the scores against the stored ones -> out/rr27_speed.jsonl
"""
import json, math, os, sys, time, urllib.request
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "cuad-retrieval"))
import retr, head, own27
OUT = os.path.join(H, "out"); FIRST = 30; SC = os.path.join(OUT, "rr27_scores.json")
SYSTEM = "You check passages of contracts. Answer with one word: Yes or No."


VARIANT = os.environ.get("RR27_VARIANT", "")      # "short": no system prompt, the clause's name only (speed test)
if VARIANT:
    SC = os.path.join(OUT, f"rr27_scores_{VARIANT}.json")


def prompt_short(t, q):
    name = retr.short(q).split(":")[0]
    return f"<|im_start|>user\n{t}\n\nIs this about \"{name}\"? Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt(t, q):
    if VARIANT == "short":
        return prompt_short(t, q)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nPassage:\n{t}\n\nDoes this passage contain a clause of this kind?\n{retr.short(q)}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def first_stage():
    """Held-out contracts with c['S'][question text] = first-stage scores of its passages."""
    C = own27.sets()["held"]; layer = json.load(open(os.path.join(OUT, "own27_pick.json")))["layer_chosen_on_practice"]; li = own27.PICK.index(layer)
    qs = sorted({it["text"] for c in C for it in c["items"]}); n = sum(len(c["texts"]) for c in C)
    m = json.load(open(os.path.join(OUT, "own27", "pick", "held", "meta.json"))); L, d = m["shape"]["l_out"]
    X = np.memmap(os.path.join(OUT, "own27", "pick", "held", "l_out.f32"), dtype=np.float32, mode="r").reshape(-1, L, d); nz = torch.nn.functional.normalize
    P = torch.tensor(np.array(X[:n, li])); Q = torch.tensor(np.array(X[n:, li])); P, Q = nz(P - P.mean(0), dim=-1), nz(Q - Q.mean(0), dim=-1); at = 0
    for c in C:
        p = P[at:at + len(c["texts"])]; at += len(c["texts"]); c["S"] = {q: p @ Q[i] for i, q in enumerate(qs)}; c["set"] = "held"
    return C, layer


def yes_no(text):
    body = {"prompt": text, "n_predict": 1, "temperature": 0, "n_probs": 20, "cache_prompt": True}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{os.environ.get('RR27_PORT', '8082')}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    top = r["completion_probabilities"][0]; top = top.get("top_logprobs") or top.get("probs"); y = n = 0.0
    for t in top:
        w = t.get("token", t.get("tok_str", "")).strip().lower(); p = math.exp(t["logprob"]) if "logprob" in t else t["prob"]
        if w.startswith("yes"):
            y += p
        elif w.startswith("no"):
            n += p
    return math.log(y + 1e-6) - math.log(n + 1e-6)


def ask(limit=None):
    C, _ = first_stage(); S = json.load(open(SC)) if os.path.exists(SC) else {}; todo = []
    for c in C:
        for it in c["items"]:
            for pi in c["S"][it["text"]].argsort(descending=True)[:FIRST].tolist():
                todo.append((c["ci"], pi, it["text"], c["texts"][pi]))
    if os.environ.get("RR27_EVERY"):                      # a fixed part of the questions: every n-th contract
        n = int(os.environ["RR27_EVERY"]); keep = {c["ci"] for c in C[::n]}; todo = [x for x in todo if x[0] in keep]
    todo = sorted(set(todo)); todo = [x for x in todo if f"{x[0]}|{x[1]}|{x[2]}" not in S][:int(limit) if limit else None]; t = time.time(); print(len(todo), "pairs to ask", flush=True)
    for k, (ci, pi, q, text) in enumerate(todo):
        S[f"{ci}|{pi}|{q}"] = yes_no(prompt(text, q))
        if k % 500 == 499 or k == len(todo) - 1:
            json.dump(S, open(SC, "w")); print(k + 1, round(time.time() - t), "s", flush=True)
    json.dump({"pairs": len(todo), "seconds": round(time.time() - t, 1)}, open(os.path.join(OUT, f"rr27_time{'_' + VARIANT if VARIANT else ''}.json"), "w"))


def score():
    C, layer = first_stage(); S = json.load(open(SC))

    def both(c, it):
        s = c["S"][it["text"]]; order = s.argsort(descending=True).tolist(); out = torch.empty(len(order))
        for r, pi in enumerate(order):
            out[pi] = 1000 + S[f"{c['ci']}|{pi}|{it['text']}"] if r < FIRST else -r
        return out
    if os.environ.get("RR27_EVERY"):
        C = C[::int(os.environ["RR27_EVERY"])]
    rep = {"layer": layer, "first": FIRST, "variant": VARIANT or "full", "contracts": len(C), "own_states": head.measure(C, "held", lambda c, it: c["S"][it["text"]]), "own_states_then_own_yes_no": head.measure(C, "held", both)}
    tf = os.path.join(OUT, f"rr27_time{'_' + VARIANT if VARIANT else ''}.json")
    if os.path.exists(tf):
        rep["time"] = json.load(open(tf))
    json.dump(rep, open(os.path.join(OUT, f"rr27{'-' + VARIANT if VARIANT else ''}{'-every' + os.environ['RR27_EVERY'] if os.environ.get('RR27_EVERY') else ''}.json"), "w"), indent=1); print(json.dumps(rep))


def speed(tag):
    from concurrent.futures import ThreadPoolExecutor
    C, _ = first_stage(); S = json.load(open(SC)); Q = [(c, it) for c in C for it in c["items"]]; Q = Q[::max(1, len(Q) // 60)][:60]; T = []; D = []
    with ThreadPoolExecutor(FIRST) as ex:
        for k, (c, it) in enumerate(Q):
            top = c["S"][it["text"]].argsort(descending=True)[:FIRST].tolist(); t = time.time()
            got = list(ex.map(lambda pi: yes_no(prompt(c["texts"][pi], it["text"])), top))
            if k:                                              # the first question warms the server up
                T.append(time.time() - t); D += [abs(g - S[f"{c['ci']}|{pi}|{it['text']}"]) for g, pi in zip(got, top)]
    T.sort(); D.sort(); r = {"tag": tag, "questions": len(T), "passages": FIRST, "mean_s": round(sum(T) / len(T), 2), "median_s": round(T[len(T) // 2], 2), "worst_s": round(T[-1], 2),
                   "score_diff_median": round(D[len(D) // 2], 3), "score_diff_p95": round(D[int(len(D) * .95)], 3)}
    open(os.path.join(OUT, "rr27_speed.jsonl"), "a").write(json.dumps(r) + "\n"); print(json.dumps(r))


if __name__ == "__main__":
    a = sys.argv[1:]; {"ask": lambda: ask(*a[1:]), "score": score, "speed": lambda: speed(a[1])}[a[0]]()
