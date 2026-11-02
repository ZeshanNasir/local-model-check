# ai-local: mission and boundaries

Read this before changing anything. It applies to every contributor, reviewer and AI agent.

## Mission

The simplest reliable way for engineers on Apple Silicon Macs to evaluate local AI models for routine engineering work. Local use may reduce cloud-model usage and cost, but savings depend on adoption and workload and are not promised.

This is a small engineering utility. It is not a general AI platform, a framework or a model leaderboard.

## Supported runtime: strict boundary

- **Ollama models that run on its MLX runner, on Apple Silicon macOS only.**
- Models come from the official Ollama library through `ollama pull`. The supported list is `MEASURED` in `labbench/lab.py`. Today: `qwen3.6:35b-mlx`, `gemma4:26b-mlx`, `qwen3.8:27b-mlx`, `flint:40b-a2b-qat-int4`.
- A model joins the list only after its results are published in `RESULTS.md`.
- Out of scope unless the owner explicitly decides otherwise: llama.cpp and other runtimes, external or remote inference servers, hosted-model fallbacks, runtime adapters.

This repository is the public benchmark tool and nothing else. Other directories or repositories on the owner's machines may also be called ai-local; do not assume they share code, configuration or scope with this one.

## Intended experience

Clone, run `./lab`, pick a supported model, approve any download, read one table. No configuration. A model is never downloaded without a typed `y`. Ollama is never installed by this project.

## Context

The reference machine is a MacBook Pro M4 Pro with 48 GB. Other Apple Silicon engineering Macs are a possible future target.

`./lab` runs at an 8,192-token context. The models advertise 262,144 tokens. Do not state that the advertised maximum works until it has been run and its memory behaviour and stability recorded on the reference machine. `RESULTS.md` records what was actually tried.

## Evidence and claims

- Keep measured results apart from assumptions and from what a model advertises.
- Report speed, tasks passed, hardware, runtime and method.
- "Tasks passed" counts 15 small synthetic tasks. It is not general model accuracy.
- Do not claim universal performance, flawless long-context use or guaranteed savings.

## Engineering rules

- Smallest change that solves a demonstrated problem. Prefer Ollama's own features and the existing code.
- Standard library only. Tests need no model and no network. Mock downloads and inference when testing control flow.
- Check the source and the final diff. A passing test run is not proof of behaviour.
- No new repositories, frameworks, routers, sidecars, speculative features or unrelated cleanup.
- Change nothing outside this repository as part of this work.

## Map of the repository

| Path | What it is |
| :--- | :--- |
| `lab` | Launcher. Checks Python, then runs `python3 -m labbench`. |
| `labbench/lab.py` | The whole user flow: checks, picker, download consent, run, table, saved report. `MEASURED` is the supported list. |
| `labbench/bench.py` | Speed: one cold run, then warm runs. |
| `labbench/workloads.py` | The 15 tasks: 6 log questions, 8 document questions, 1 code fix. |
| `labbench/scoring.py` | Checks answers with code, not with another model. |
| `labbench/ollama.py` | Minimal client for the local Ollama server. `is_loopback()` lets `lab.py` refuse a non-local `OLLAMA_HOST`. |
| `labbench/env.py` | Records the machine and Ollama version with each run. |
| `labbench/stats.py` | Median and spread. |
| `data/synthetic/` | Task inputs. All invented, no real data. |
| `tests/` | Deterministic tests. Run `python3 -m unittest discover -s tests`. |
| `RESULTS.md` | The published measurements and their method. |
| `benchmarks/results/` | Raw files behind `RESULTS.md`. Includes long-prompt, coding-agent and embedding runs that `./lab` no longer performs. Kept as evidence, not reproducible with `./lab`. |
| `results/` | Where `./lab` saves your own runs. Not tracked by git. |

Flow: `lab` → `labbench/lab.py` → `bench.py` and `workloads.py` (via `scoring.py`) → `ollama.py` → local Ollama.

## Status

Stable. Change it only for a specific, authorized task or a concrete problem reported by someone who used it. Until then, leave it alone.
