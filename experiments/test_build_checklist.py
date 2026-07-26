from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_checklist.py"
SPEC = importlib.util.spec_from_file_location("build_checklist", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildChecklistTests(unittest.TestCase):
    def test_every_response_slot_is_filled(self) -> None:
        marker = "Type your response here"
        template = "\n".join([marker] * len(MODULE.ANSWERS))
        output = MODULE.fill_checklist(template)
        self.assertNotIn(marker, output)
        self.assertEqual(len(output.splitlines()), len(MODULE.ANSWERS))

    def test_template_drift_is_detected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.fill_checklist("Type your response here")


if __name__ == "__main__":
    unittest.main()
