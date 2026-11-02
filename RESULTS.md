# Results

Measured on 2026-11-02 on an Apple M4 Pro with 48 GB, macOS 27.0.1, Ollama 0.40.13, on AC power. Each model ran every task three times through `./check`. Raw files: [`benchmarks/results/2026-11-02`](benchmarks/results/2026-11-02/), one file per `./check` run.

| Model | Speed | Tasks passed | Verdict |
| :--- | ---: | ---: | :--- |
| `qwen3.6:35b-mlx` | 67 tok/s | 37/45 | fast, missed the code fix and a log task in every run, a document task in 2 of 3 runs |
| `gemma4:26b-mlx` | 56 tok/s | 42/45 | fast, missed the code fix in every run |
| `qwen3.8:27b-mlx` | 28 tok/s | 42/45 | slow, missed a log task in every run |
| `flint:40b-a2b-qat-int4` | 98 tok/s | 33/45 | fast, missed 3 log tasks and a document task in every run |

The previous measurement of 2026-10-09 (Ollama 0.40.1, [raw files](benchmarks/results/2026-10-09/)) gave 68, 60 and 26 tok/s and 43, 42 and 42 of 45 tasks for the first three models. Speeds moved by 5 tok/s or less. Only `qwen3.6:35b-mlx` changed in task results.

## What else was found

From the 2026-10-09 run (Ollama 0.40.1). These tests are not part of `./check`.

- **Long prompts cost minutes.** At about 67,000 tokens the first word took 137 s (`gemma4:26b-mlx`), 150 s (`qwen3.6:35b-mlx`) and 704 s (`qwen3.8:27b-mlx`). Swap use grew by about 1.8 GB and 3.2 GB for the last two. Only `gemma4:26b-mlx` was tried at about 136,000 tokens: it recalled the planted fact after 388 s and swap grew by about 1.8 GB.
- **The advertised context was not tested.** The models advertise 262,144 tokens. Nothing here shows that length works on this Mac. `./check` runs at 8,192 tokens.
- **Letting a model run the tests helps.** `gemma4:26b-mlx` failed the code fix when answering in one go, but fixed it in three of three sessions in the OpenCode coding agent, where it could run the tests.
- **Local is not automatically private.** Ollama only used this machine. OpenCode, the agent around it, also connected to outside servers whose purpose was not identified.

The long-prompt and agent tests are not part of `./check` any more. Their raw files are kept with the others.

## How it is measured

- **Speed:** median decode speed over 3 warm runs of a 1,000-token prompt, from Ollama's own timings. Thinking is off.
- **Tasks passed:** 15 synthetic tasks, checked by code, not by another model. 6 ask whether a log supports a stated cause, 8 ask questions about short documents (some stale or conflicting), 1 asks for a bug fix that must pass its tests. This counts passes on small tasks; it is not general model accuracy.
- **Fast** means 40 tokens per second or more. That is a local rule of thumb, not a standard.

One machine, one day, small synthetic tasks. This does not predict how a model performs on your real work.
