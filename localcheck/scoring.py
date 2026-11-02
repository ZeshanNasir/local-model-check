"""Deterministic scorers. A task counts as passed only if every check passes."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_json(text):
    """Return the first JSON object in a model reply, or None. Models sometimes wrap it in prose."""
    text = text.strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except ValueError:
            continue
    return None


def score_log(reply, expect):
    data = parse_json(reply)
    if data is None:
        return {"passed": False, "why": "no_json"}
    supports = data.get("supports_conclusion")
    lines = {int(x) for x in data.get("evidence_lines", []) if str(x).lstrip("-").isdigit()}
    if supports is not expect["supports"]:
        return {"passed": False, "why": "wrong_abstention" if not expect["supports"] else "false_abstention"}
    if expect["supports"]:
        if not set(expect["evidence"]) <= lines:
            return {"passed": False, "why": "missing_evidence"}
        if len(lines - set(expect["evidence"])) > 2:
            return {"passed": False, "why": "too_many_extra_lines"}
    return {"passed": True, "why": ""}


def score_doc(reply, expect):
    data = parse_json(reply)
    if data is None:
        return {"passed": False, "why": "no_json"}
    if data.get("status") != expect["status"]:
        return {"passed": False, "why": f"status_{data.get('status')}"}
    sources = set(data.get("sources", []) or [])
    if expect["status"] != "insufficient" and not set(expect["sources"]) <= sources:
        return {"passed": False, "why": "wrong_sources"}
    if expect["status"] == "answered" and expect["contains"] not in str(data.get("answer", "")).lower():
        return {"passed": False, "why": "answer_missing_fact"}
    return {"passed": True, "why": ""}


# Model-written code runs here, so it never runs unconfined: no network, no system services, writes only inside the project.
_SANDBOX = '(version 1)(allow default)(deny network*)(deny mach-lookup)(deny file-write* (require-not (subpath (param "WORK"))))'


def run_tests(repo_dir, timeout=60):
    """Run a project's unit tests under macOS sandbox-exec. Returns (passed, output tail)."""
    if not shutil.which("sandbox-exec"):
        return False, "sandbox-exec not found"
    work = str(Path(repo_dir).resolve())
    env = {"PATH": "/usr/bin:/bin", "HOME": work, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(["sandbox-exec", "-D", f"WORK={work}", "-p", _SANDBOX, sys.executable, "-m", "unittest", "-q"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return p.returncode == 0, (p.stdout + p.stderr)[-1500:]


def score_patch(template_repo, files):
    """Apply {path: content} to a temporary copy of the repo and run its tests.

    Also checks scope: only files that already existed may change, and not the tests."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp).resolve() / "repo"
        shutil.copytree(template_repo, work)
        changed = []
        for rel, content in files.items():
            target = (work / rel).resolve()
            if work.resolve() not in target.parents or not target.exists():
                return {"passed": False, "why": "path_outside_repo_or_new_file", "changed": [rel]}
            if target.read_text() != content:
                changed.append(rel)
                target.write_text(content)
        if any(c.startswith("tests/") for c in changed):
            return {"passed": False, "why": "edited_tests", "changed": changed}
        ok, tail = run_tests(work)
        return {"passed": ok, "why": "" if ok else "tests_fail", "changed": changed, "test_tail": "" if ok else tail}
