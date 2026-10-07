import tempfile
import unittest
from pathlib import Path

from dppm.data import Dataset, permute, write_jsonl


class DataTests(unittest.TestCase):
    def test_permutation_preserves_answer_identity(self):
        for key in ["persona/benchmark/1", "prefeval/travel_transportation/0/choice/300"]:
            choices, label = permute(["correct", "other1", "other2", "other3"], key)
            self.assertEqual(choices[label], "correct")
            self.assertEqual(
                (choices, label), permute(["correct", "other1", "other2", "other3"], key)
            )

    def test_missing_history_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            write_jsonl(p / "sessions.jsonl", [])
            write_jsonl(
                p / "examples.jsonl",
                [dict(id="x", options=["a", "b"], label=0, sessions=["missing"])],
            )
            with self.assertRaises(ValueError):
                Dataset(p)
