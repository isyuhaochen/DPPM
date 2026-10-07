"""Compare a complete rerun to an archived run without exporting private source paths."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {r["id"]: r for r in rows}
    if len(rows) != len(indexed) or not rows:
        raise ValueError("Empty or duplicate prediction IDs")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference, actual = read(args.reference), read(args.actual)
    if reference.keys() != actual.keys():
        raise ValueError("Different question coverage")
    labels = sum(reference[k]["label"] != actual[k]["label"] for k in reference)
    predictions = sum(reference[k]["prediction"] != actual[k]["prediction"] for k in reference)
    errors = []
    for key in reference:
        a, b = reference[key]["logits"], actual[key]["logits"]
        if len(a) != len(b):
            raise ValueError("Different option counts")
        errors.extend(abs(x - y) for x, y in zip(a, b))
    correct = sum(r["prediction"] == r["label"] for r in actual.values())
    report = dict(
        count=len(actual),
        correct=correct,
        accuracy=100 * correct / len(actual),
        label_mismatches=labels,
        prediction_mismatches=predictions,
        max_absolute_logit_difference=max(errors),
        identical_logits=all(x == 0 for x in errors),
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        actual_sha256=hashlib.sha256(Path(args.actual).read_bytes()).hexdigest(),
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if labels or predictions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
