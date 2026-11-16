#!/usr/bin/env python3
"""The released, general Contrastive-LM head on our contract search. Nothing here is fitted on contracts.
    python clm.py embed    Qwen3-8B (frozen, loaded in 8 bits to fit the card), last-token state, unit length, for every
                           passage of the 62 held-out contracts and every question                 -> out/clm_vecs.pt
    python clm.py score    ranks passages three ways and scores them like head.py                  -> out/head.jsonl
        raw        cosine in the encoder's own space (the authors' `clm-raw` ablation)
        head       their state head on the question, their action head on each passage, cosine (CLM-v0.1-8B as released)
      each with the question in two forms (the full CUAD question; the short "clause: description" form), since the
      head was trained on prose states and nothing says which suits it; both are reported, neither is chosen.
Encoder input is the bare text, as their server sends it. Their encoder runs unquantized under vLLM; ours is 8-bit."""
import importlib.util, json, os, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "..", "cuad-retrieval"); sys.path.insert(0, E1)
import retr, head
OUT = os.path.join(H, "out"); M = os.path.join(H, "..", "models")
FORMS = {"full": lambda q: q, "short": retr.short}


def embed():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    C = [c for c in head.setup() if c["set"] == "held"]
    tok = AutoTokenizer.from_pretrained(os.path.join(M, "Qwen3-8B"), padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(os.path.join(M, "Qwen3-8B"), quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0}).eval()

    def enc(texts, batch=16):
        out = [None] * len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for b in range(0, len(order), batch):
            ids = order[b:b + batch]; x = tok([texts[i] for i in ids], return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                h = m(**x, output_hidden_states=True).hidden_states[-1][:, -1].float()
            for i, v in zip(ids, torch.nn.functional.normalize(h, dim=-1).cpu()):
                out[i] = v
            if b % 800 == 0:
                print("embedded", b, "of", len(texts), flush=True)
        return torch.stack(out)

    qs = sorted({it["text"] for c in C for it in c["items"]}); V = {"P": {c["ci"]: enc(c["texts"]) for c in C}, "q": qs}
    for f, fn in FORMS.items():
        V[f] = enc([fn(q) for q in qs])
    torch.save(V, os.path.join(OUT, "clm_vecs.pt")); print("saved", sum(len(p) for p in V["P"].values()), "passages", len(qs), "questions")


def score():
    spec = importlib.util.spec_from_file_location("clm_heads", os.path.join(H, "..", "leads", "CLM", "src", "clm", "heads.py")); hd = importlib.util.module_from_spec(spec); spec.loader.exec_module(hd)
    ck = torch.load(os.path.join(M, "CLM-v0.1-8B", "CLM_v0.1-8B.pt"), map_location="cpu", weights_only=True); cfg = dict(ck["cfg"])
    kw = dict(width=cfg["width"], depth=cfg["depth"], proj=ck.get("projection_dim", cfg.get("projection_dim", 512)), activation=cfg.get("activation", "gelu"),
              layernorm=cfg.get("layernorm", False), residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", 4096))
    sh, ah = hd.make_head(**kw), hd.make_head(**kw); sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"]); sh.eval(); ah.eval()
    V = torch.load(os.path.join(OUT, "clm_vecs.pt")); qi = {q: i for i, q in enumerate(V["q"])}; C = [c for c in head.setup() if c["set"] == "held"]; nz = torch.nn.functional.normalize
    with torch.no_grad():
        PA = {ci: nz(ah(p), dim=-1) for ci, p in V["P"].items()}; QS = {f: nz(sh(V[f]), dim=-1) for f in FORMS}
    for f in FORMS:
        head.report(f"Qwen3-8B (8-bit) raw cosine, {f} question", C, lambda c, it: V["P"][c["ci"]] @ V[f][qi[it["text"]]], {"fitted_on_contracts": False})
        head.report(f"Qwen3-8B (8-bit) + released CLM-v0.1-8B head, {f} question", C, lambda c, it: PA[c["ci"]] @ QS[f][qi[it["text"]]], {"fitted_on_contracts": False, "head_cfg": {k: str(v) for k, v in cfg.items()}})


if __name__ == "__main__":
    {"embed": embed, "score": score}[sys.argv[1]]()
