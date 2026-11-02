import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from labbench import bench, lab, ollama, scoring, stats, workloads

REPO = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "coding" / "repo"
FIXED = '''def apply_discount(price_cents: int, percent: int) -> int:
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return (price_cents * (100 - percent) + 50) // 100
'''


class Scoring(unittest.TestCase):
    def test_json_is_found_inside_prose(self):
        self.assertEqual(scoring.parse_json('Sure: {"a": 1} done'), {"a": 1})
        self.assertIsNone(scoring.parse_json("no json here"))

    def test_log_abstention_and_evidence(self):
        want = {"supports": True, "evidence": [4]}
        ok = json.dumps({"supports_conclusion": True, "evidence_lines": [4, 5]})
        self.assertTrue(scoring.score_log(ok, want)["passed"])
        self.assertEqual(scoring.score_log(json.dumps({"supports_conclusion": True, "evidence_lines": [1]}), want)["why"], "missing_evidence")
        self.assertEqual(scoring.score_log(json.dumps({"supports_conclusion": False, "evidence_lines": []}), want)["why"], "false_abstention")
        none = {"supports": False, "evidence": []}
        self.assertEqual(scoring.score_log(json.dumps({"supports_conclusion": True, "evidence_lines": [2]}), none)["why"], "wrong_abstention")

    def test_doc_scoring(self):
        want = {"status": "answered", "sources": ["a"], "contains": "14"}
        self.assertTrue(scoring.score_doc(json.dumps({"status": "answered", "answer": "14 days", "sources": ["a"]}), want)["passed"])
        self.assertEqual(scoring.score_doc(json.dumps({"status": "answered", "answer": "7 days", "sources": ["a"]}), want)["why"], "answer_missing_fact")
        self.assertEqual(scoring.score_doc(json.dumps({"status": "insufficient"}), want)["why"], "status_insufficient")
        self.assertTrue(scoring.score_doc(json.dumps({"status": "insufficient"}), {"status": "insufficient"})["passed"])

    def test_the_shipped_coding_task_fails_until_fixed(self):
        self.assertFalse(scoring.run_tests(REPO)[0])
        self.assertTrue(scoring.score_patch(REPO, {"calc/pricing.py": '"""x"""\n' + FIXED})["passed"])

    def test_patch_scope_is_enforced(self):
        self.assertEqual(scoring.score_patch(REPO, {"tests/test_pricing.py": "pass\n"})["why"], "edited_tests")
        self.assertEqual(scoring.score_patch(REPO, {"../escape.py": "x"})["why"], "path_outside_repo_or_new_file")
        self.assertEqual(scoring.score_patch(REPO, {"calc/new.py": "x"})["why"], "path_outside_repo_or_new_file")
        self.assertEqual(scoring.score_patch(REPO, {"calc/pricing.py": "def apply_discount(p, c):\n    return p\n"})["why"], "tests_fail")


class Measurement(unittest.TestCase):
    def test_summary_and_spread(self):
        self.assertEqual(stats.summarize([1, 2, 3])["median"], 2)
        self.assertEqual(stats.spread_pct([9, 10, 11]), 20.0)
        self.assertIsNone(stats.spread_pct([5]))

    def test_filler_size_and_cache_busting(self):
        a, b = bench.filler(1000, "x"), bench.filler(1000, "y")
        self.assertGreater(sum(map(len, a)), 2700)
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1:], b[1:])

    def test_workloads_wiring_with_a_canned_model(self):
        def canned(model, prompt, **kw):
            if "Logs:" in prompt:
                return ollama.Generation(text=json.dumps({"supports_conclusion": False, "evidence_lines": []}), decode_tokens=5, decode_s=1)
            if "Question:" in prompt and "Use \"insufficient\"" in prompt:
                return ollama.Generation(text=json.dumps({"status": "insufficient"}), decode_tokens=5, decode_s=1)
            return ollama.Generation(text=json.dumps({"path": "calc/pricing.py", "content": '"""x"""\n' + FIXED}), decode_tokens=5, decode_s=1)
        with mock.patch.object(ollama, "generate", canned):
            rows = workloads.run_workloads("m:1")
        by = {w: [r["passed"] for r in rows if r["workload"] == w] for w in ("logs", "docs", "code")}
        self.assertEqual(by["code"], [True])
        self.assertEqual(sum(by["logs"]), 3)   # only the three abstention cases pass with this canned answer
        self.assertEqual(sum(by["docs"]), 2)   # only the two "insufficient" cases pass


class Data(unittest.TestCase):
    def test_every_question_topic_exists_and_logs_are_numbered_consistently(self):
        docs = workloads.load_docs()
        for q in workloads.load_questions():
            self.assertTrue(set(q["topic"]) <= set(docs), q["id"])
            self.assertTrue(set(q["expect"].get("sources", [])) <= set(docs), q["id"])
        for case in workloads.load_logs():
            self.assertTrue(all(1 <= n <= len(case["lines"]) for n in case["expect"]["evidence"]), case["id"])


class Lab(unittest.TestCase):
    def test_loopback_detection(self):
        for host, ok in (("127.0.0.1:11434", True), ("http://localhost:11434", True), ("10.0.0.5:11434", False), ("https://example.com", False)):
            with mock.patch.dict("os.environ", {"OLLAMA_HOST": host}):
                self.assertEqual(ollama.is_loopback(), ok, host)

    def test_remote_server_is_refused(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "10.0.0.5:11434"}), \
             mock.patch.object(lab.sys, "platform", "darwin"), mock.patch.object(lab.platform, "machine", return_value="arm64"):
            self.assertIn("not this Mac", lab.preflight())

    def test_embedding_models_are_skipped(self):
        installed = [{"name": "gemma4:26b-mlx", "family": ""}, {"name": "embeddinggemma-2:740m", "family": "embedding_gemma2"}]
        self.assertEqual(lab.chat_models(installed), ["gemma4:26b-mlx"])

    def test_verdict_and_table(self):
        good = {"model": "a:1", "decode_tok_s": 68.3, "passed": 15, "total": 15, "missed": []}
        slow = {"model": "b:1", "decode_tok_s": 26.1, "passed": 13, "total": 15, "missed": ["logs", "code"]}
        self.assertEqual(lab.verdict(good), "fast, all tasks passed")
        self.assertEqual(lab.verdict(slow), "slow, missed 1 log task, code fix")
        self.assertIn("68 tok/s", lab.table([good, slow]))
        self.assertIn("Tasks passed", lab.table([good]))

    def test_choose_returns_the_picked_model(self):
        with mock.patch("builtins.print"):
            self.assertEqual(lab.choose([], ask=lambda _: "2"), "gemma4:26b-mlx")
            self.assertEqual(lab.choose([], ask=lambda _: str(len(lab.MEASURED))), lab.MEASURED[-1])
            for answer in ("", "0", str(len(lab.MEASURED) + 1), "x"):
                self.assertIsNone(lab.choose([], ask=lambda _, a=answer: a))
            self.assertIsNone(lab.choose([], ask=mock.Mock(side_effect=EOFError)))

    def test_download_needs_an_explicit_yes(self):
        with mock.patch.object(lab.subprocess, "run") as run:
            for answer in ("", "n", "yes please", " "):
                self.assertFalse(lab.pull("m:1", ask=lambda _, a=answer: a))
            self.assertFalse(lab.pull("m:1", ask=mock.Mock(side_effect=EOFError)))
            run.assert_not_called()
            run.return_value.returncode = 0
            self.assertTrue(lab.pull("m:1", ask=lambda _: " Y "))
            run.assert_called_once_with(["ollama", "pull", "m:1"])
            run.return_value.returncode = 1
            self.assertFalse(lab.pull("m:1", ask=lambda _: "y"))

    def _run_main(self, argv, installed, pick=None, pulled=False):
        """Run lab.main with every outside effect mocked. Returns (exit code, calls in order, results dir exists)."""
        calls = mock.Mock()
        calls.pull.return_value = pulled
        calls.test.return_value = {"model": "m", "decode_tok_s": 60, "passed": 15, "total": 15, "missed": []}
        snap = {"chip": "c", "memory_gib": 1, "macos": "27", "ollama_version": "x", "installed_models": [{"name": n} for n in installed]}
        with tempfile.TemporaryDirectory() as root, mock.patch.object(lab, "ROOT", Path(root)), \
             mock.patch.object(lab, "preflight", return_value=None), mock.patch.object(lab.env, "snapshot", return_value=snap), \
             mock.patch.object(lab, "choose", return_value=pick), mock.patch.object(lab, "pull", calls.pull), \
             mock.patch.object(lab, "test", calls.test), mock.patch("sys.stderr"), mock.patch("builtins.print"):
            code = lab.main(argv)
            return code, [c[0] for c in calls.mock_calls], (Path(root) / "results").exists()

    def test_approved_download_runs_the_benchmark_only_after_the_pull_succeeds(self):
        code, calls, saved = self._run_main([], installed=[], pick="gemma4:26b-mlx", pulled=True)
        self.assertEqual((code, calls, saved), (0, ["pull", "test"], True))

    def test_declined_or_failed_download_runs_nothing_and_saves_nothing(self):
        code, calls, saved = self._run_main([], installed=[], pick="gemma4:26b-mlx", pulled=False)  # pull() is False for no, EOF and a failed pull
        self.assertEqual((code, calls, saved), (1, ["pull"], False))

    def test_named_missing_model_uses_the_same_consent_rule(self):
        self.assertEqual(self._run_main(["gemma4:26b-mlx"], installed=[], pulled=True)[1], ["pull", "test"])
        self.assertEqual(self._run_main(["gemma4:26b-mlx"], installed=[], pulled=False)[1], ["pull"])

    def test_model_outside_the_measured_list_is_rejected_even_when_installed(self):
        self.assertEqual(self._run_main(["other:1"], installed=[], pulled=True), (1, [], False))
        self.assertEqual(self._run_main(["other:1"], installed=["other:1"]), (1, [], False))
        self.assertEqual(self._run_main(["gemma4:26b-mlx", "other:1"], installed=[], pulled=True), (1, [], False))  # nothing pulled before the check

    def test_quitting_the_picker_runs_nothing_and_saves_nothing(self):
        self.assertEqual(self._run_main([], installed=[], pick=None), (0, [], False))

    def test_installed_model_is_tested_without_a_download(self):
        self.assertEqual(self._run_main([], installed=["gemma4:26b-mlx"], pick="gemma4:26b-mlx"), (0, ["test"], True))


if __name__ == "__main__":
    unittest.main()
