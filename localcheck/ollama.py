"""Minimal Ollama HTTP client (standard library only) that records timing for every request."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


def base_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return host if host.startswith("http") else f"http://{host}"


def is_loopback():
    return urllib.parse.urlsplit(base_url()).hostname in ("127.0.0.1", "localhost", "::1")


@dataclass
class Generation:
    text: str = ""
    ttft_s: float = 0.0  # request sent -> first streamed token (answer or reasoning)
    wall_s: float = 0.0
    load_s: float = 0.0  # model load time reported by the server
    prompt_tokens: int = 0
    prompt_s: float = 0.0
    decode_tokens: int = 0
    decode_s: float = 0.0
    error: str = ""

    @property
    def prompt_tok_s(self):
        return self.prompt_tokens / self.prompt_s if self.prompt_s else 0.0

    @property
    def decode_tok_s(self):
        return self.decode_tokens / self.decode_s if self.decode_s else 0.0


def _request(path, body, timeout):
    req = urllib.request.Request(base_url() + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def generate(model, prompt, *, num_ctx=8192, num_predict=256, seed=1, think=False,
             keep_alive="10m", timeout=900):
    """Stream one completion. temperature is 0 so repeated runs are comparable."""
    body = {"model": model, "prompt": prompt, "stream": True, "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0, "seed": seed}}
    if think is not None:
        body["think"] = think
    out = Generation()
    start = time.perf_counter()
    try:
        try:
            resp = _request("/api/generate", body, timeout)
        except urllib.error.HTTPError as e:
            if think is not None and e.code == 400:  # model without a thinking switch
                body.pop("think")
                resp = _request("/api/generate", body, timeout)
            else:
                raise
        with resp:
            for line in resp:
                chunk = json.loads(line)
                if chunk.get("error"):
                    out.error = chunk["error"]
                    break
                piece, thought = chunk.get("response", ""), chunk.get("thinking", "")
                if (piece or thought) and not out.ttft_s:
                    out.ttft_s = time.perf_counter() - start
                out.text += piece
                if chunk.get("done"):
                    out.load_s = chunk.get("load_duration", 0) / 1e9
                    out.prompt_tokens = chunk.get("prompt_eval_count", 0)
                    out.prompt_s = chunk.get("prompt_eval_duration", 0) / 1e9
                    out.decode_tokens = chunk.get("eval_count", 0)
                    out.decode_s = chunk.get("eval_duration", 0) / 1e9
    except Exception as e:  # timeouts and connection errors are results, not crashes
        out.error = f"{type(e).__name__}: {e}"
    out.wall_s = time.perf_counter() - start
    return out


def get(path, timeout=10):
    with urllib.request.urlopen(base_url() + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=120):
    with _request(path, body, timeout) as r:
        return json.loads(r.read())


def loaded():
    """Models currently resident, with the server's own memory figure for each."""
    return get("/api/ps").get("models", [])


def unload(model):
    post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)


def unload_all():
    for m in loaded():
        unload(m["name"])

