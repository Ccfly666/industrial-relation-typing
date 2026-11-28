"""Orchestration checks: no training or benchmark data are needed."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_scope_suite import plan, preflight


class ScopeSuiteTests(unittest.TestCase):
    def test_scope_label_and_projector_bindings(self):
        commands = plan("E", "pool.npz", "labels", "names.npz", "scopes", "cuda", "projectors")
        self.assertEqual(len(commands), 20)
        outputs = set()
        for command in commands:
            def value(flag):
                return command[command.index(flag) + 1]
            stem = Path(value("--out")).stem
            scope = stem.split("_", 1)[1]
            self.assertEqual(Path(value("--fit-labels")).name, f"E__{scope}__fit_labels.npz")
            self.assertEqual(Path(value("--projector")).name, f"{stem}_projector.npz")
            self.assertNotIn("outer_truth", " ".join(command))
            if "--inner" in command:
                self.assertEqual(Path(value("--validation-labels")).name, f"E__{scope}__validation_labels.npz")
            else:
                self.assertNotIn("--validation-labels", command)
            outputs.add(value("--out"))
        self.assertEqual(len(outputs), 20)
        self.assertEqual(sum("--inner" not in c for c in commands), 5)

    def test_preflight_checks_later_scope_before_execution(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inp = root / "input.npz"
            inp.touch()
            commands = [["python", "builder.py", "--pool", str(inp), "--out", str(root / "first.npz")],
                        ["python", "builder.py", "--pool", str(root / "missing.npz"), "--out", str(root / "last.npz")]]
            with self.assertRaises(FileNotFoundError):
                preflight(commands)
            self.assertFalse((root / "first.npz").exists())
            commands[1][3] = str(inp)
            preflight(commands)
            (root / "last.npz").touch()
            with self.assertRaises(FileExistsError):
                preflight(commands)


if __name__ == "__main__":
    unittest.main()
