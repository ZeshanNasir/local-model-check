"""The three synthetic engineering workloads: log analysis, document Q&A, a small code fix."""
import json
from pathlib import Path

from . import ollama
from .scoring import parse_json, run_tests, score_doc, score_log, score_patch

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic"

LOG_PROMPT = """You are analysing operational logs. Lines are numbered from 1.
Answer the question using only the logs. Reply with JSON only:
{{"finding": "<one sentence>", "evidence_lines": [<line numbers that support the finding>], "supports_conclusion": <true or false>}}
If the logs do not show a cause, set supports_conclusion to false and evidence_lines to [].

Question: {question}

Logs:
{logs}
"""

DOC_PROMPT = """Answer the question using only the documents below. Reply with JSON only:
{{"status": "answered" | "insufficient" | "conflict", "answer": "<short answer>", "sources": ["<document ids used>"]}}
Use "insufficient" if the documents do not contain the answer. Use "conflict" if documents disagree and nothing
says which is current. If one document is clearly newer and replaces another, answer from the newer one.

{docs}

Question: {question}
"""

CODE_PROMPT = """The unit tests in this small Python project fail. Fix the bug with the smallest change.
Do not edit the tests. Reply with JSON only: {{"path": "<file to change>", "content": "<the complete new file>"}}

Project files:
{files}

Test output:
{tests}
"""


def load_logs():
    return json.loads((DATA / "logs.json").read_text())


def load_questions():
    return json.loads((DATA / "questions.json").read_text())


def load_docs():
    return {p.stem: p.read_text() for p in sorted((DATA / "docs").glob("*.md"))}


def log_prompt(case):
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(case["lines"], 1))
    return LOG_PROMPT.format(question=case["question"], logs=numbered)


def doc_prompt(question, docs):
    blob = "\n\n".join(f"[{name}]\n{text}" for name, text in docs.items())
    return DOC_PROMPT.format(docs=blob, question=question)


def code_prompt():
    repo = DATA / "coding" / "repo"
    files = "\n\n".join(f"--- {p.relative_to(repo)} ---\n{p.read_text()}" for p in sorted(repo.rglob("*.py")) if p.read_text().strip())
    return CODE_PROMPT.format(files=files, tests=run_tests(repo)[1])


def _score(fn, *args):
    """Model output is untrusted: a malformed reply fails the task instead of stopping the run."""
    try:
        return fn(*args)
    except (TypeError, ValueError, AttributeError, KeyError):
        return {"passed": False, "why": "malformed_reply"}


def run_workloads(model, *, num_ctx=8192, think=False, repeats=1):
    """Run every case; return one record per (case, repeat) with score and timing."""
    out = []
    docs = load_docs()
    for rep in range(repeats):
        for case in load_logs():
            g = ollama.generate(model, log_prompt(case), num_ctx=num_ctx, num_predict=400, think=think)
            out.append({"workload": "logs", "case": case["id"], "rep": rep, **_timing(g), **_score(score_log, g.text, case["expect"])})
        for q in load_questions():
            g = ollama.generate(model, doc_prompt(q["question"], docs), num_ctx=num_ctx, num_predict=300, think=think)
            out.append({"workload": "docs", "case": q["id"], "rep": rep, **_timing(g), **_score(score_doc, g.text, q["expect"])})
        g = ollama.generate(model, code_prompt(), num_ctx=num_ctx, num_predict=700, think=think)
        data = parse_json(g.text) or {}
        verdict = _score(score_patch, DATA / "coding" / "repo", {data["path"]: data["content"]}) if "path" in data and "content" in data else {"passed": False, "why": "no_patch"}
        out.append({"workload": "code", "case": "pricing-fix", "rep": rep, **_timing(g), "passed": verdict["passed"], "why": verdict["why"]})
    return out


def _timing(g):
    return {"wall_s": round(g.wall_s, 2), "decode_tokens": g.decode_tokens, "decode_tok_s": round(g.decode_tok_s, 1), "error": g.error}
