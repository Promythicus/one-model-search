#!/usr/bin/env python3
"""Sharpening the embedding search for contracts with the model frozen (after Contrastive-LM/CLM).
    python head.py base        the embedder as it is                                    (rung 0)
    python head.py linear      a closed-form add-on: bilinear ridge on practice pairs    (rung 2; no training loop)
    python head.py train       a small trained add-on on both sides, contrastive loss,   (rung 3; trains ~2M numbers,
                               other-contract negatives first, same-contract ones second  the embedder untouched)
    python head.py types       the honesty check for `linear`: CUAD asks the same 41 kinds of question of every
                               contract, so the add-on may only have learnt those 41. Fit it on half the kinds
                               (even-numbered), score held-out contracts on the kinds it saw and on those it never saw.
Each prints and stores (out/head.jsonl) hit/span at 6k and 12k on the held-out contracts, for the ranking alone and
with the stock reranker re-ordering its first 30. Same split as rr.py (seed 3: 40 practice, 62 held out); of the 40,
8 are set aside to choose settings (ridge strength, mixing weight, number of epochs)."""
import json, os, random, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "..", "cuad-retrieval"); sys.path.insert(0, E1)
import bench, retr, rr
OUT = os.path.join(H, "out"); SPEC = "500"


def setup():
    Dt = retr.data(); V, Q, qi = retr.vectors("qwen06", SPEC, Dt); practice, held = rr.split()
    val = set(sorted(practice)[::5]); C = []; at = 0
    for ci, (title, text, qs) in enumerate(Dt):
        ch, texts = retr.passages(title, text, SPEC); P = torch.tensor(V[at:at + len(ch)]); at += len(ch); items = []
        for q, gold in qs:
            pos = torch.tensor([any(a < ge and gs < b for gs, ge in gold) for a, b in ch])
            items.append({"q": torch.tensor(Q["short"][qi[q]].astype(np.float32)), "pos": pos, "gold": gold, "text": q, "k": int(qi[q])})
        C.append({"ci": ci, "P": P, "ch": ch, "texts": texts, "items": items, "set": "held" if ci in held else ("val" if ci in val else "fit")})
    return C


def measure(C, which, score, scorer=None, keep=None):
    """score(c, item) -> a score per passage of the contract. With `scorer`, the stock reranker re-orders the first 30."""
    out = {"hit6k": 0, "span6k": 0, "hit12k": 0, "span12k": 0}; n = ns = 0
    for c in C:
        if c["set"] != which:
            continue
        for it in c["items"]:
            if keep and not keep(it):
                continue
            order = score(c, it).argsort(descending=True).tolist()
            if scorer:
                first = order[:30]; s = scorer(retr.short(it["text"]), [c["texts"][i] for i in first]); order = [first[i] for i in sorted(range(len(first)), key=lambda i: -s[i])] + order[30:]
            n += 1; ns += len(it["gold"])
            for name, top in (("6k", 12), ("12k", 24)):
                got = [c["ch"][i] for i in order[:top]]; heldv = [any(a < ge and gs < b for a, b in got) for gs, ge in it["gold"]]
                out["hit" + name] += any(heldv); out["span" + name] += sum(heldv)
    return {k: round(v / (n if k.startswith("hit") else ns), 4) for k, v in out.items()}


def report(tag, C, score, extra=None):
    r = {"tag": tag, "held_out": measure(C, "held", score)}
    torch.cuda.empty_cache(); scorer = retr.reranker("qwenrr"); r["held_out_with_reranker"] = measure(C, "held", score, scorer)
    if extra:
        r.update(extra)
    open(os.path.join(OUT, "head.jsonl"), "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)


def base():
    C = setup(); report("embedder as it is", C, lambda c, it: c["P"] @ it["q"])


def linear():
    C = setup()

    def pairs(sets):                                  # for each question: its positives and its 30 best-ranked passages
        Qs, Ps, Y = [], [], []
        for c in C:
            if c["set"] in sets:
                for it in c["items"]:
                    top = (c["P"] @ it["q"]).topk(min(30, len(c["P"]))).indices; ix = torch.unique(torch.cat([top, it["pos"].nonzero().flatten()]))
                    Qs.append(it["q"].expand(len(ix), -1)); Ps.append(c["P"][ix]); Y.append(it["pos"][ix].float())
        return torch.cat(Qs).double().cuda(), torch.cat(Ps).double().cuda(), torch.cat(Y).double().cuda()

    def solve(sets, lam):
        q, p, y = pairs(sets); K = (q @ q.T) * (p @ p.T); a = torch.linalg.solve(K + lam * K.diag().mean() * torch.eye(len(K), device="cuda", dtype=K.dtype), y - y.mean())
        return (q * a[:, None]).T @ p                                   # M [1024, 1024]: add-on score = q M p

    best = None
    for lam in (0.01, 0.1, 1.0):
        M = solve({"fit"}, lam).float().cpu()
        for beta in (0.25, 0.5, 1.0, 2.0):
            v = measure(C, "val", lambda c, it: c["P"] @ it["q"] + beta * (c["P"] @ (M.T @ it["q"])))
            if best is None or v["span6k"] > best[0]:
                best = (v["span6k"], lam, beta)
    _, lam, beta = best; M = solve({"fit", "val"}, lam).float().cpu(); torch.save({"M": M, "beta": beta, "lam": lam}, os.path.join(OUT, "linear.pt"))
    report("closed-form bilinear add-on", C, lambda c, it: c["P"] @ it["q"] + beta * (c["P"] @ (M.T @ it["q"])), {"lam": lam, "beta": beta, "chosen_on_val_span6k": best[0]})


def types():
    C = setup(); S = torch.load(os.path.join(OUT, "linear.pt")); lam, beta = S["lam"], S["beta"]; Qs, Ps, Y = [], [], []
    for c in C:
        if c["set"] != "held":
            for it in c["items"]:
                if it["k"] % 2 == 0:
                    top = (c["P"] @ it["q"]).topk(min(30, len(c["P"]))).indices; ix = torch.unique(torch.cat([top, it["pos"].nonzero().flatten()]))
                    Qs.append(it["q"].expand(len(ix), -1)); Ps.append(c["P"][ix]); Y.append(it["pos"][ix].float())
    q, p, y = torch.cat(Qs).double().cuda(), torch.cat(Ps).double().cuda(), torch.cat(Y).double().cuda()
    K = (q @ q.T) * (p @ p.T); a = torch.linalg.solve(K + lam * K.diag().mean() * torch.eye(len(K), device="cuda", dtype=K.dtype), y - y.mean())
    M = ((q * a[:, None]).T @ p).float().cpu(); plain = lambda c, it: c["P"] @ it["q"]; plus = lambda c, it: c["P"] @ it["q"] + beta * (c["P"] @ (M.T @ it["q"]))
    r = {"tag": "add-on fitted on even-numbered question kinds only", "kinds": len({it["k"] for c in C for it in c["items"]}), "lam": lam, "beta": beta}
    for name, keep in (("kinds_seen", lambda it: it["k"] % 2 == 0), ("kinds_never_seen", lambda it: it["k"] % 2 == 1)):
        r[name] = {"embedder": measure(C, "held", plain, keep=keep), "with_add_on": measure(C, "held", plus, keep=keep)}
    open(os.path.join(OUT, "types.json"), "w").write(json.dumps(r, indent=1)); print(json.dumps(r))


def train():
    C = setup(); torch.manual_seed(0); d = 1024
    A = torch.zeros(d, d, device="cuda", requires_grad=True); B = torch.zeros(d, d, device="cuda", requires_grad=True)       # q' = q + A q, p' = p + B p; zero = the embedder as it is
    opt = torch.optim.AdamW([A, B], lr=2e-5, weight_decay=0.01); nz = torch.nn.functional.normalize
    fit = [c for c in C if c["set"] == "fit"]; rnd = random.Random(0)
    score = lambda c, it: (nz(c["P"].cuda() + c["P"].cuda() @ B.T, dim=-1) @ nz(it["q"].cuda() + A @ it["q"].cuda(), dim=-1)).detach().cpu()
    hist = []; best = (measure(C, "val", score)["span6k"], 0, A.detach().clone(), B.detach().clone())
    for epoch in range(1, 41):
        hard = epoch > 10; order = [(c, it) for c in fit for it in c["items"] if it["pos"].any()]; rnd.shuffle(order)
        for c, it in order:
            P = c["P"].cuda(); q = nz(it["q"].cuda() + A @ it["q"].cuda(), dim=-1); pos = it["pos"].cuda()
            if not hard:                                               # stage 1: negatives from another contract
                o = rnd.choice(fit)["P"].cuda(); P = torch.cat([P[pos], o]); pos = torch.cat([torch.ones(int(pos.sum()), dtype=torch.bool, device="cuda"), torch.zeros(len(o), dtype=torch.bool, device="cuda")])
            s = nz(P + P @ B.T, dim=-1) @ q / 0.05
            loss = torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0)
            opt.zero_grad(); loss.backward(); opt.step()
        v = measure(C, "val", score)["span6k"]; hist.append(v)
        if v > best[0]:
            best = (v, epoch, A.detach().clone(), B.detach().clone())
        print("epoch", epoch, "val span6k", v, flush=True)
    _, epoch, A, B = best; torch.save({"A": A.cpu(), "B": B.cpu(), "epoch": epoch}, os.path.join(OUT, "trained.pt"))
    report("trained add-on (contrastive, 2 x 1024^2)", C, score, {"epochs_kept": epoch, "val_span6k_by_epoch": hist})


if __name__ == "__main__":
    {"base": base, "linear": linear, "train": train, "types": types}[sys.argv[1]]()
