"""Throughput and latency measurement."""
import time

from . import ollama
from .stats import spread_pct, summarize

_SERVICES = ["billing", "catalog", "gateway", "orders", "search", "payments", "inventory", "auth"]


def filler(target_tokens, salt, chars_per_token=2.8):
    """Deterministic synthetic log-like text, about target_tokens long.

    chars_per_token differs between tokenizers; use calibrate() for a real model. The salt changes the
    first line so a server-side prompt cache cannot reuse an earlier run."""
    lines, chars, i = [f"run-id {salt}"], 0, 0
    while chars < target_tokens * chars_per_token:
        svc = _SERVICES[i % len(_SERVICES)]
        line = f"entry {i}: service {svc} handled {100 + (i * 37) % 900} requests with {(i * 11) % 7} retries and p95 {20 + (i * 13) % 180} ms"
        lines.append(line)
        chars += len(line) + 1
        i += 1
    return lines


def calibrate(model):
    """Characters per token for this model's tokenizer on the filler text, from one 1-token request."""
    text = "\n".join(filler(2000, "calibration"))
    g = ollama.generate(model, text, num_ctx=4096, num_predict=1, think=False)
    return len(text) / g.prompt_tokens if g.prompt_tokens else 2.8


def throughput(model, *, runs=3, num_ctx=8192, prompt_tokens=1000, num_predict=256):
    """One cold run (model unloaded first) then `runs` warm runs with cache-busting prompts."""
    cpt = calibrate(model)  # loads the model, so unload again afterwards for a true cold start
    ollama.unload_all()
    time.sleep(3)
    results = []
    for n in range(runs + 1):
        text = "\n".join(filler(prompt_tokens, f"{model}-{n}", cpt))
        prompt = f"Summarise the main operational patterns in this log in about 150 words.\n\n{text}"
        g = ollama.generate(model, prompt, num_ctx=num_ctx, num_predict=num_predict, think=False)
        results.append({"cold": n == 0, "error": g.error, "load_s": round(g.load_s, 2), "ttft_s": round(g.ttft_s, 2),
                        "prompt_tokens": g.prompt_tokens, "prompt_tok_s": round(g.prompt_tok_s, 1),
                        "decode_tokens": g.decode_tokens, "decode_tok_s": round(g.decode_tok_s, 1), "wall_s": round(g.wall_s, 2)})
    warm = [r for r in results[1:] if not r["error"]]
    return {
        "model": model, "num_ctx": num_ctx, "runs": results,
        "warm_summary": {k: summarize([r[k] for r in warm]) for k in ("ttft_s", "prompt_tok_s", "decode_tok_s", "wall_s")},
        "decode_spread_pct": spread_pct([r["decode_tok_s"] for r in warm]),
    }

