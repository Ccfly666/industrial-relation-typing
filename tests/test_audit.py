import unittest, tempfile, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit, fetch_sources


class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_four_relations_direction_duplicates_conflicts_missing(self):
        rec = [
            dict(source_uri="s" + str(i), target_uri="t", relation=r)
            for i, r in enumerate(audit.LABELS)
        ]
        good, issues, sm = audit.clean(
            rec, {r["source_uri"]: {} for r in rec}, {"t": {}}
        )
        self.assertEqual(len(good), 4)
        self.assertEqual(audit.REVERSE[">"], "<")
        self.assertEqual(audit.REVERSE["<"], ">")
        good, issues, sm = audit.clean(
            rec
            + [
                rec[0],
                dict(rec[1], relation="~"),
                dict(source_uri="missing", target_uri="t", relation="="),
            ],
            {r["source_uri"]: {} for r in rec},
            {"t": {}},
        )
        self.assertEqual(sm["same_label_duplicates"], 1)
        self.assertEqual(sm["conflict_pairs"], 1)
        self.assertEqual(sm["other_invalid"], 1)

    def test_namespaces_escape_uri_and_multiple_parents(self):
        p = self.p / "a.rdf"
        p.write_text(
            '<r:RDF xmlns:r="'
            + audit.RDF
            + '" xmlns:o="'
            + audit.OWL
            + '" xmlns:s="'
            + audit.RDFS
            + '" xml:base="http://x/ont"><o:Class r:about="#a"><s:label>A &amp; B</s:label><s:subClassOf r:resource="#b"/><s:subClassOf r:resource="#c%20d"/></o:Class><o:Class r:about="#b"/><o:Class r:about="#c%20d"/></r:RDF>'
        )
        c = audit.concepts(p)
        self.assertEqual(c["http://x/ont#a"]["name"], "A & B")
        self.assertEqual(c["http://x/ont#a"]["depth"], 1)
        self.assertEqual(len(c["http://x/ont#a"]["parents"]), 2)

    def test_cycle_empty_malformed(self):
        p = self.p / "a.rdf"
        p.write_text(
            '<r:RDF xmlns:r="'
            + audit.RDF
            + '" xmlns:o="'
            + audit.OWL
            + '" xmlns:s="'
            + audit.RDFS
            + '"><o:Class r:about="http://a"><s:subClassOf r:resource="http://a"/></o:Class></r:RDF>'
        )
        with self.assertRaises(ValueError):
            audit.concepts(p)
        p.write_text('<r:RDF xmlns:r="' + audit.RDF + '"/>')
        self.assertEqual(audit.concepts(p), {})
        self.assertEqual(audit.assertions(p, "x"), [])
        p.write_text("broken")
        with self.assertRaises(Exception):
            audit.concepts(p)


if __name__ == "__main__":
    unittest.main()
