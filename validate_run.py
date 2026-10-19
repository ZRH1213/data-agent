"""Check that a completed Hadoop run reconciles with its saved artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from pipeline import ROOT, RUNS, TABLES, ZIP_PATH, aggregate_scores, digest


def validate(run: Path) -> dict:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert report["status"] == status["status"] == "completed"
    assert report["task_id"] == status["task_id"] == run.name
    assert report["source_sha256"] == digest(ZIP_PATH)
    assert len(report["hadoop_jobs"]) == 9
    for job in report["hadoop_jobs"]:
        assert (ROOT / job["log"]).is_file(), job

    clean_hashes = []
    counts = {}
    for table in TABLES:
        part = report["cleaning"][table]
        actions = part["actions"]
        before = report["counts"]["before"][table]["total"]
        after = report["counts"]["after"][table]["total"]
        assert before == sum(actions.values()), table
        assert after == actions.get("keep", 0), table
        expected_reasons = (actions.get("deduplicate", 0) + actions.get("quarantine", 0)
                            + part["reasons"].get("normalized", 0))
        assert sum(part["reasons"].values()) == expected_reasons, table
        for reason, accepted in part.get("repair_reasons", {}).items():
            assert accepted <= part.get("repair_attempts", {}).get(reason, 0), (table, reason)
        clean = run / "clean" / f"{table}.dat"
        quarantine = run / "quarantine" / f"{table}.jsonl"
        assert digest(clean) == part["clean_sha256"], table
        with clean.open(encoding="utf-8") as source:
            assert sum(1 for _ in source) == after, table
        with quarantine.open(encoding="utf-8") as source:
            assert sum(1 for _ in source) == actions.get("quarantine", 0), table
        clean_hashes.append(part["clean_sha256"])
        counts[table] = {"raw": before, **actions}
    version = hashlib.sha256("".join(clean_hashes).encode()).hexdigest()
    assert report["data_version"] == version
    assert report["scores"]["before"] == aggregate_scores(report["counts"]["before"])
    assert report["scores"]["after"] == aggregate_scores(report["counts"]["after"])
    assert sum(report["time_split"]["counts"].values()) == counts["ratings"]["keep"]
    assert report["time_split"]["T1"] < report["time_split"]["T2"]
    return {"task_id": run.name, "status": "validated", "jobs": 9, "counts": counts,
            "data_version": version}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", help="Task directory name under artifacts/runs")
    args = parser.parse_args()
    print(json.dumps(validate(RUNS / args.task_id), ensure_ascii=False, indent=2))
