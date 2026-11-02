"""./check: which local model should I use on this Mac?

Tests one of the MLX-format models in MEASURED, through Ollama on this Mac, for speed and tasks passed,
prints one table and saves it. Downloads a model only after you type y; never talks to another machine.
"""
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import bench, env, ollama, workloads

ROOT = Path(__file__).resolve().parent.parent
FAST_TOK_S = 40  # a local rule of thumb, not a standard
MEASURED = ("qwen3.6:35b-mlx", "gemma4:26b-mlx", "qwen3.8:27b-mlx", "flint:40b-a2b-qat-int4")  # official Ollama library tags, measured in RESULTS.md. The only models ./check accepts.


def preflight():
    """The first problem that stops a run, as a plain sentence, or None."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return f"local-model-check runs on macOS with Apple Silicon only (this is {sys.platform}/{platform.machine()})."
    if not ollama.is_loopback():
        return f"OLLAMA_HOST points at {ollama.base_url()}, not this Mac. Unset it to test local models."
    if not shutil.which("ollama"):
        return "Ollama is not installed. Get it from https://ollama.com/download, open it once, then run ./check again."
    try:
        ollama.get("/api/version", timeout=3)
    except Exception:
        return "Ollama is installed but not running. Open the Ollama app, then run ./check again."
    return None


def chat_models(installed):
    return [m["name"] for m in installed if "embed" not in m["name"].lower() and "embed" not in (m.get("family") or "").lower()]


def pull(model, ask=input):
    """Download one model only after an explicit y. Anything else, including no terminal, declines."""
    try:
        answer = ask(f"{model} is not installed. Download it with `ollama pull {model}`? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() == "y" and subprocess.run(["ollama", "pull", model]).returncode == 0


def choose(installed, ask=input):
    """List the measured models and return the one picked, or None."""
    for i, name in enumerate(MEASURED, 1):
        print(f"  {i}. {name:<18} {'installed' if name in installed else 'not installed'}")
    try:
        pick = ask(f"Pick 1-{len(MEASURED)} (Enter to quit): ").strip()
    except EOFError:
        return None
    return MEASURED[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(MEASURED) else None


def test(model):
    speed = bench.throughput(model, runs=3)
    tasks = workloads.run_workloads(model)
    ollama.unload_all()
    return {"model": model, "decode_tok_s": speed["warm_summary"]["decode_tok_s"]["median"],
            "passed": sum(t["passed"] for t in tasks), "total": len(tasks),
            "missed": [t["workload"] for t in tasks if not t["passed"]], "throughput": speed, "tasks": tasks}


def verdict(r):
    speed = "fast" if (r["decode_tok_s"] or 0) >= FAST_TOK_S else "slow"
    if not r["missed"]:
        return f"{speed}, all tasks passed"
    parts = [f"{r['missed'].count(w)} {name}{'s' if r['missed'].count(w) > 1 else ''}"
             for w, name in (("logs", "log task"), ("docs", "document task")) if w in r["missed"]]
    if "code" in r["missed"]:
        parts.append("code fix")
    return f"{speed}, missed {', '.join(parts)}"


def table(results):
    rows = [("Model", "Speed", "Tasks passed", "Verdict")] + [
        (r["model"], f"{r['decode_tok_s'] or 0:.0f} tok/s", f"{r['passed']}/{r['total']}", verdict(r)) for r in results]
    width = [max(len(row[i]) for row in rows) for i in range(3)]
    return "\n".join(f"{a:<{width[0]}}  {b:>{width[1]}}  {c:>{width[2]}}  {d}" for a, b, c, d in rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: ./check [MODEL]\n\nTests a model for speed and tasks passed. Without MODEL it shows a picker.\nMODEL is one of: " + ", ".join(MEASURED))
        return 0
    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 1
    snap = env.snapshot()
    installed = chat_models(snap["installed_models"])
    models = argv or [choose(installed)]
    if models == [None]:
        return 0
    unsupported = [m for m in models if m not in MEASURED]
    if unsupported:
        print(f"Not supported: {', '.join(unsupported)}. Supported: {', '.join(MEASURED)}", file=sys.stderr)
        return 1
    for m in models:
        if m not in installed and not pull(m):
            print(f"Not installed: {m}", file=sys.stderr)
            return 1

    machine = f"{snap['chip']}, {snap['memory_gib']} GB, macOS {snap['macos']}, Ollama {snap['ollama_version']}"
    print(machine)
    results = []
    for model in models:
        print(f"Testing {model} (a few minutes)...", flush=True)
        results.append(test(model))

    text = table(results)
    snap["installed_models"] = [m for m in snap["installed_models"] if m["name"] in models]  # keep only what was tested
    out = ROOT / "results" / datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(f"# local-model-check\n\n{machine}\n\n```\n{text}\n```\n")
    (out / "results.json").write_text(json.dumps({"environment": snap, "results": results}, indent=1) + "\n")
    print(f"\n{text}\n\nSaved: {out.relative_to(ROOT)}/")
    return 0
