import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "provider_templates/codex/model-catalogs/deepseek-v4.json"
NOTICE = REPO_ROOT / "provider_templates/codex/model-catalogs/deepseek-v4.NOTICE.md"


class PublicProvenanceTests(unittest.TestCase):
    def test_deepseek_catalog_is_valid_json_with_adjacent_derivation_notice(self):
        document = json.loads(CATALOG.read_text())
        self.assertEqual(
            [model["slug"] for model in document["models"]],
            ["deepseek-flash", "deepseek-v4-pro"],
        )
        notice = NOTICE.read_text()
        self.assertIn("modified derivative", notice)
        self.assertIn("deepseek-flash", notice)
        self.assertIn("deepseek-v4-flash", notice)
        self.assertIn("deepseek-v4-pro", notice)
        self.assertIn("Copyright 2025 OpenAI", notice)
        self.assertIn("Apache-2.0", notice)
