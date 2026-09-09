import tempfile
import unittest
from pathlib import Path

from benchmark.freeze import collect_hashes, verify_lock


class FreezeTests(unittest.TestCase):
    def test_committed_fixtures_match_the_lock(self):
        self.assertEqual(verify_lock(), [])

    def test_interpreter_cache_artifacts_are_not_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fixtures" / "package" / "__pycache__").mkdir(parents=True)
            (root / "tasks" / "prompts" / "__pycache__").mkdir(parents=True)
            (root / "fixtures" / "package" / "source.py").write_text("value = 1\n")
            (root / "fixtures" / "package" / "__pycache__" / "source.cpython-312.pyc").write_bytes(b"cache")
            (root / "tasks" / "prompts" / "task.md").write_text("task\n")
            (root / "tasks" / "prompts" / "__pycache__" / "task.pyc").write_bytes(b"cache")

            hashes = collect_hashes(root)
            self.assertEqual(
                set(hashes),
                {
                    "fixtures/package/source.py",
                    "tasks/prompts/task.md",
                },
            )
