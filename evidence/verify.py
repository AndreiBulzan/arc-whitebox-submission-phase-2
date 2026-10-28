"""Check the release's recorded arithmetic; this is not a predictor benchmark."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def verify_table(name, summaries):
    groups = defaultdict(list)
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw, flops, adjusted = (float(row[k]) for k in ("raw_final_mse", "metered_flops", "adjusted_score"))
            assert int(row["failed"]) == 0
            assert math.isclose(adjusted, raw * max(0.1, flops / 2**41), rel_tol=2e-12)
            groups[row["variant"]].append((raw, flops, adjusted))
    for item in summaries:
        values = groups[item["variant"]]
        expected = (item["raw_final_mse"], item["mean_flops"], item["adjusted_score"])
        for i in range(3):
            assert math.isclose(sum(row[i] for row in values) / len(values), expected[i], rel_tol=2e-12)
    return sum(map(len, groups.values()))


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    read = lambda name: json.loads((ROOT / name).read_text())
    count = verify_table("research/pc2/scores.csv", read("research/pc2/RESULTS.json")["variants"])
    count += verify_table("evidence/mainline_rows.csv", read("evidence/mainline_history.json")["records"])
    count += verify_table("evidence/effective_birth_rows.csv", read("evidence/effective_birth.json")["records"])
    with (ROOT / "evidence/graded_public_rows.csv").open(newline="") as stream:
        displayed = list(csv.DictReader(stream))
    assert len(displayed) == 100
    for row in displayed:
        raw = float(row["raw_final_mse_displayed"])
        calculated = raw * max(0.1, float(row["metered_flops"]) / 2**41)
        # Public per-row scores were rounded in the displayed grader record.
        assert math.isclose(calculated, float(row["adjusted_score_displayed"]), rel_tol=1e-3)
        assert int(row["failed"]) == 0
    print(f"PASS: file hashes, {count} historical score rows and 100 rounded public grade rows.")
    print("This checks saved evidence, not a new inference or performance result.")


if __name__ == "__main__":
    main()
