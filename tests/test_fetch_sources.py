import unittest, tempfile, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_sources


class Fetch(unittest.TestCase):
    def test_zip_traversal_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            z = base / "x.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                fetch_sources.safe_extract(z, base / "out")
            p = base / "snapshot"
            p.write_bytes(b"a")
            self.assertEqual(
                fetch_sources.obtain(base / "copy", fetch_sources.sha(p), p),
                "LOCAL_SNAPSHOT_VERIFIED",
            )
            with self.assertRaises(ValueError):
                fetch_sources.obtain(base / "copy", "0" * 64, p)


if __name__ == "__main__":
    unittest.main()
