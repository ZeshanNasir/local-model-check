# local-model-check: mission and boundaries

Read this before changing anything. It applies to every contributor, reviewer and AI agent.

## Mission

The simplest reliable way for engineers on Apple Silicon Macs to evaluate local AI models for routine engineering work. Local use may reduce cloud-model usage and cost, but savings depend on adoption and workload and are not promised.

This is a small engineering utility. It is not a general AI platform, a framework or a model leaderboard.

## Supported runtime: strict boundary

- **Ollama models that run on its MLX runner, on Apple Silicon macOS only.**
- Models come from the official Ollama library through `ollama pull`. The supported list is `MEASURED` in `localcheck/cli.py`. Today: `qwen3.6:35b-mlx`, `gemma4:26b-mlx`, `qwen3.8:27b-mlx`, `flint:40b-a2b-qat-int4`.
- A model joins the list only after its results are published in `RESULTS.md`.
- Out of scope unless the owner explicitly decides otherwise: llama.cpp and other runtimes, external or remote inference servers, hosted-model fallbacks, runtime adapters.

## Intended experience

Clone, run `./check`, pick a supported model, approve any download, read one table. No configuration. A model is never downloaded without a typed `y`. Ollama is never installed by this project.

## Context

The reference machine is a MacBook Pro M4 Pro with 48 GB. Other Apple Silicon engineering Macs are a possible future target.

`./check` runs at an 8,192-token context. The models advertise 262,144 tokens. Do not state that the advertised maximum works until it has been run and its memory behaviour and stability recorded on the reference machine. `RESULTS.md` records what was actually tried.

## Evidence and claims

- Keep measured results apart from assumptions and from what a model advertises.
- Report speed, tasks passed, hardware, runtime and method.
- "Tasks passed" counts 15 small synthetic tasks. It is not general model accuracy.
- Do not claim universal performance, flawless long-context use or guaranteed savings.

## Engineering rules

- Smallest change that solves a demonstrated problem. Prefer Ollama's own features and the existing code.
- Standard library only. A test fails if anything else is imported. Tests need no model and no network. Mock downloads and inference when testing control flow.
- Model replies and model-written code are untrusted. A malformed reply fails its task and never stops the run. The code-fix task runs under `sandbox-exec` with no network and no writes outside a temporary copy of the project. Do not weaken either.
- The tool talks only to Ollama on this Mac, and a saved run holds only timings, the tested model, and the Mac model, chip, memory, macOS, Python and Ollama versions, power source, free memory and swap use. No hostnames, usernames, serial numbers, paths or other installed models.
- Check the source and the final diff. A passing test run is not proof of behaviour.
- No new repositories, frameworks, routers, sidecars, speculative features or unrelated cleanup.
- Change nothing outside this repository as part of this work.

## Map of the repository

| Path | What it is |
| :--- | :--- |
| `check` | Launcher. Checks Python, then runs `python3 -m localcheck`. |
| `localcheck/cli.py` | The whole user flow: checks, picker, download consent, run, table, saved report. `MEASURED` is the supported list. |
| `localcheck/bench.py` | Speed: one cold run, then warm runs. |
| `localcheck/workloads.py` | The 15 tasks: 6 log questions, 8 document questions, 1 code fix. |
| `localcheck/scoring.py` | Checks answers with code, not with another model. |
| `localcheck/ollama.py` | Minimal client for the local Ollama server. `is_loopback()` lets `cli.py` refuse a non-local `OLLAMA_HOST`. |
| `localcheck/env.py` | Records the machine and Ollama version with each run. |
| `localcheck/stats.py` | Median and spread. |
| `data/synthetic/` | Task inputs. All invented, no real data. |
| `tests/` | Deterministic tests. Run `python3 -m unittest discover -s tests`. |
| `RESULTS.md` | The published measurements and their method. |
| `benchmarks/results/` | Raw files behind `RESULTS.md`. Includes long-prompt, coding-agent and embedding runs that `./check` does not perform. Kept as evidence, not reproducible with `./check`. |
| `results/` | Where `./check` saves your own runs. Not tracked by git. |

Flow: `check` → `localcheck/cli.py` → `bench.py` and `workloads.py` (via `scoring.py`) → `ollama.py` → local Ollama.

## Status

Stable. Change it only for a specific, authorized task or a concrete problem reported by someone who used it. Until then, leave it alone.
