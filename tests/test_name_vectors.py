import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_name_vectors import pack, prepare


class NameVectors(unittest.TestCase):
    def test_prepare_pack_and_reject_reordered_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rdf = root / "source.rdf"
            rdf.write_text(
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
                'xmlns:owl="http://www.w3.org/2002/07/owl#">'
                '<owl:Class rdf:about="urn:a"><rdfs:label>A</rdfs:label></owl:Class>'
                '<owl:Class rdf:about="urn:b"><rdfs:label>B</rdfs:label></owl:Class>'
                "</rdf:RDF>",
                encoding="utf-8",
            )
            digest = hashlib.sha256(rdf.read_bytes()).hexdigest()
            keys = np.array([f"{digest}|urn:b", f"{digest}|urn:a"])
            pool = root / "pool.npz"
            np.savez(pool, source_keys=keys, target_keys=keys[:1])
            names = root / "names"
            prepare(pool, rdf, rdf, names)
            self.assertEqual(
                json.loads((names / "source_names.json").read_text()), ["B", "A"]
            )
            source, target = root / "s.npz", root / "t.npz"
            vectors = np.arange(768, dtype=np.float32).reshape(2, 384)
            np.savez(source, names=np.array(["B", "A"]), embeddings=vectors)
            np.savez(target, names=np.array(["B"]), embeddings=vectors[:1])
            pack(pool, names, source, target, root / "joined.npz")
            with np.load(root / "joined.npz", allow_pickle=False) as result:
                np.testing.assert_array_equal(result["source_embeddings"], vectors)
                np.testing.assert_array_equal(result["source_keys"], keys)
            np.savez(source, names=np.array(["A", "B"]), embeddings=vectors)
            with self.assertRaisesRegex(ValueError, "out of order"):
                pack(pool, names, source, target, root / "bad.npz")

    def test_reject_wrong_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rdf = root / "source.rdf"
            rdf.write_text(
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>',
                encoding="utf-8",
            )
            np.savez(
                root / "pool.npz",
                source_keys=np.array(["wrong|urn:a"]),
                target_keys=np.array([]),
            )
            with self.assertRaisesRegex(ValueError, "snapshot"):
                prepare(root / "pool.npz", rdf, rdf, root / "names")


if __name__ == "__main__":
    unittest.main()
