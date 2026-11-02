# local-model-check

Check how fast a local AI model runs on your Mac, and whether it can handle small engineering tasks.

[![CI](https://github.com/ZeshanNasir/local-model-check/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeshanNasir/local-model-check/actions/workflows/ci.yml)

```sh
git clone https://github.com/ZeshanNasir/local-model-check.git
cd local-model-check
./check
```

Pick a model. A few minutes later you get one table, and the same table is saved in `results/`:

```
Model              Speed  Tasks passed  Verdict
gemma4:26b-mlx  56 tok/s         14/15  fast, missed code fix
```

Speed is how fast the model writes, in tokens per second (a token is a few characters). Tasks passed counts 15 small synthetic tasks, scored by code and not by another model: 6 log questions, 8 questions about short documents, and 1 bug fix.

## Built for review

- **Python standard library only.** Nothing to install or audit. A test fails if anything else is imported.
- **Stays on your Mac.** It talks only to Ollama on this Mac and stops if `OLLAMA_HOST` points elsewhere. No telemetry, no accounts.
- **Downloads only after you type `y`.** It never installs Ollama or anything else.
- **Model-written code is confined.** The bug-fix task runs the model's patch under macOS `sandbox-exec`: no network, and no writes outside a temporary copy of the project. A test proves it.
- **Saves little.** `results/` (ignored by Git) holds the table and timings, the Mac model, chip, memory, macOS, Python and Ollama versions, power source, free memory and swap use, and only the model you tested. No hostnames, usernames, serial numbers or paths.
- **Starts each test cold.** It unloads the models Ollama has loaded.

You need an Apple Silicon Mac, Python 3.10 or newer, and Ollama installed and running.

## Models

`./check` accepts only the models listed in [RESULTS.md](RESULTS.md), which also gives each one's measured speed and tasks passed. All are official Ollama library tags that ran on Ollama's MLX runner. `ollama show MODEL` prints the minimum Ollama version a model needs. To skip the picker: `./check gemma4:26b-mlx`.

## What the results do not show

They do not show that a local model can replace a cloud model, rank models in general, or promise any saving. The test uses an 8,192-token context; the 262,144 tokens the models advertise are not tested. One Mac and small synthetic tasks: check a model on your own work before you rely on it.

## Contributing and security

Read [AGENTS.md](AGENTS.md) first. `python3 -m unittest discover -s tests` needs no model and no network. Report security issues as described in [SECURITY.md](SECURITY.md). MIT licence.
