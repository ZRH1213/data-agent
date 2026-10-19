"""Orchestrate real Hadoop Streaming jobs and versioned MovieLens artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "ml-1m.zip"
RUNS = ROOT / "artifacts" / "runs"
IMAGE = "movielens-hadoop-python:3.4.1"
TABLES = ("users", "movies", "ratings")
RULE_VERSION = "movielens-clean-v1"
SCORE_VERSION = "five-dimensions-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_hadoop_runtime() -> None:
    try:
        daemon = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                capture_output=True, text=True, timeout=20)
        if daemon.returncode:
            raise RuntimeError(daemon.stderr.strip() or "Docker engine unavailable")
        image = subprocess.run(["docker", "image", "inspect", IMAGE],
                               capture_output=True, text=True, timeout=20)
        if image.returncode:
            build = subprocess.run(["docker", "build", "-t", IMAGE, "-f", "Dockerfile.hadoop", "."],
                                   cwd=ROOT, capture_output=True, text=True, timeout=600)
            if build.returncode:
                raise RuntimeError("Hadoop/Python image build failed: " +
                                   "\n".join(build.stderr.splitlines()[-12:]))
        probe = subprocess.run(
            ["docker", "run", "--rm", IMAGE, "bash", "-c",
             "command -v hadoop && command -v python3 && "
             "find /opt/hadoop/share/hadoop/tools/lib -name 'hadoop-streaming-*.jar' | head -n 1"],
            capture_output=True, text=True, timeout=60)
        if probe.returncode or len(probe.stdout.strip().splitlines()) < 3:
            raise RuntimeError(probe.stderr.strip() or "Hadoop/Python/Streaming jar missing")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Hadoop runtime preflight failed: {exc}") from exc


def ingest(dataset: Path, raw: Path) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset) as archive:
        for table in TABLES:
            name = f"ml-1m/{table}.dat"
            if name not in archive.namelist():
                raise FileNotFoundError(f"Archive entry missing: {name}")
            with archive.open(name) as source, (raw / f"{table}.dat").open("w", encoding="utf-8", newline="\n") as target:
                for line in source:
                    target.write(line.decode("iso-8859-1").rstrip("\r\n") + "\n")
    return {table: digest(raw / f"{table}.dat") for table in TABLES}


def in_container(path: Path) -> str:
    return "/workspace/" + path.relative_to(ROOT).as_posix()


def stream_job(name: str, input_file: Path, output_dir: Path, mapper: str,
               reducer: str, log_dir: Path) -> dict:
    """Run a Hadoop job in local MapReduce mode inside the official image."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    jar = '$(find /opt/hadoop/share/hadoop/tools/lib -name "hadoop-streaming-*.jar" | head -n 1)'
    command = (
        "set -eu; "
        f"hadoop jar {jar} "
        "-D mapreduce.framework.name=local -D fs.defaultFS=file:/// "
        "-D mapreduce.job.reduces=1 "
        f"-input {shlex.quote(in_container(input_file))} "
        f"-output {shlex.quote(in_container(output_dir))} "
        f"-mapper {shlex.quote(mapper)} -reducer {shlex.quote(reducer)}"
    )
    args = ["docker", "run", "--rm", "-v", f"{ROOT}:/workspace", "-w", "/workspace",
            IMAGE, "bash", "-c", command]
    start = utc_now()
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=900)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-12:])
        raise RuntimeError(f"Hadoop job {name} failed (exit {result.returncode}): {tail}")
    parts = sorted(output_dir.glob("part-*"))
    if not parts:
        raise RuntimeError(f"Hadoop job {name} produced no part file")
    return {"job": name, "started_at": start, "finished_at": utc_now(),
            "log": str((log_dir / f"{name}.log").relative_to(ROOT)), "part_files": len(parts)}


def output_lines(directory: Path):
    for path in sorted(directory.glob("part-*")):
        with path.open(encoding="utf-8") as source:
            yield from source


def score_table(run: Path, stage: str, table: str, users: Path, movies: Path,
                reference: int, jobs: list) -> dict:
    input_file = run / stage / f"{table}.dat"
    output_dir = run / "hadoop" / f"score-{stage}-{table}"
    mapper = (
        f"python3 /workspace/hadoop_worker.py score-map --table {table} "
        f"--users {in_container(users)} --movies {in_container(movies)} "
        f"--reference {reference}"
    )
    reducer = "python3 /workspace/hadoop_worker.py score-reduce"
    jobs.append(stream_job(f"score-{stage}-{table}", input_file, output_dir,
                           mapper, reducer, run / "logs"))
    counts = Counter()
    for line in output_lines(output_dir):
        name, value = line.rstrip("\n").split("\t", 1)
        counts[name] += int(value)
    if counts["total"] == 0:
        raise RuntimeError(f"No rows scored in {stage}/{table}")
    return dict(counts)


def clean_table(run: Path, table: str, users: Path | None, movies: Path | None,
                jobs: list) -> dict:
    input_file = run / "raw" / f"{table}.dat"
    output_dir = run / "hadoop" / f"clean-{table}"
    mapper = f"python3 /workspace/hadoop_worker.py clean-map --table {table}"
    if users:
        mapper += f" --users {in_container(users)}"
    if movies:
        mapper += f" --movies {in_container(movies)}"
    reducer = "python3 /workspace/hadoop_worker.py clean-reduce"
    jobs.append(stream_job(f"clean-{table}", input_file, output_dir,
                           mapper, reducer, run / "logs"))
    counts = Counter()
    reasons = Counter()
    repair_reasons = Counter()
    repair_attempts = Counter()
    examples = []
    example_reasons = set()
    clean_path = run / "clean" / f"{table}.dat"
    quarantine_path = run / "quarantine" / f"{table}.jsonl"
    with clean_path.open("w", encoding="utf-8", newline="\n") as clean, quarantine_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejected:
        for line in output_lines(output_dir):
            item = json.loads(line)
            counts[item["action"]] += 1
            repair_attempts.update(item.get("repairs", []))
            if item["reason"]:
                reasons[item["reason"]] += 1
                if item["reason"] not in example_reasons and len(examples) < 8:
                    examples.append({"action": item["action"], "reason": item["reason"],
                                     "raw": item["raw"][:200]})
                    example_reasons.add(item["reason"])
            if item["action"] == "keep":
                repair_reasons.update(item.get("repairs", []))
                clean.write(item["row"] + "\n")
            elif item["action"] == "quarantine":
                rejected.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"actions": dict(counts), "reasons": dict(reasons),
            "repair_attempts": dict(repair_attempts),
            "repair_reasons": dict(repair_reasons), "examples": examples,
            "clean_sha256": digest(clean_path)}


def aggregate_scores(tables: dict) -> dict:
    total = sum(item["total"] for item in tables.values())
    if not total:
        raise RuntimeError("Quality scores have no denominator")
    count = lambda field: sum(item.get(field, 0) for item in tables.values())
    pct = lambda value, denominator: round(100 * value / denominator, 4)
    ratings = tables["ratings"]
    return {
        "Accurate": pct(count("accurate_proxy"), total),
        "Complete": pct(count("complete"), total),
        "Unique": pct(total - count("duplicate_excess"), total),
        "Up-to-date": pct(ratings.get("fresh", 0), ratings["total"]),
        "Consistent": pct(count("consistent"), total),
    }


def time_split(ratings: Path) -> dict:
    with ratings.open(encoding="utf-8") as source:
        stamps = sorted(int(line.rstrip("\n").split("::")[3]) for line in source)
    if len(stamps) < 10:
        raise RuntimeError("Too few ratings to establish T1/T2")
    t1, t2 = stamps[int(len(stamps) * 0.8) - 1], stamps[int(len(stamps) * 0.9) - 1]
    if t1 >= t2:
        raise RuntimeError("T1/T2 quantiles overlap; split needs manual review")
    counts = {"train": sum(x <= t1 for x in stamps),
              "validation": sum(t1 < x <= t2 for x in stamps),
              "test": sum(x > t2 for x in stamps)}
    return {"T1": t1, "T2": t2,
            "T1_utc": datetime.fromtimestamp(t1, timezone.utc).isoformat(),
            "T2_utc": datetime.fromtimestamp(t2, timezone.utc).isoformat(),
            "method": "80th/90th chronological rating timestamp quantiles; ties stay in earlier split",
            "counts": counts}


def report_markdown(report: dict) -> str:
    before, after = report["scores"]["before"], report["scores"]["after"]
    lines = ["# MovieLens 1M iteration 1: measured quality report", "",
             f"Task: {report['task_id']}",
             f"Raw SHA-256: {report['source_sha256']}",
             f"Clean data version: {report['data_version']}",
             f"Rule versions: {report['rule_version']} / {report['score_version']}",
             f"UTC execution: {report['started_at']} to {report['finished_at']}",
             f"Hadoop image: {report['hadoop_image']}; completed jobs: {len(report['hadoop_jobs'])}",
             "", "## Five quality dimensions", "",
             "| Dimension | Before | After | Delta (points) |",
             "| --- | ---: | ---: | ---: |"]
    for name in before:
        lines.append(f"| {name} | {before[name]:.4f} | {after[name]:.4f} | {after[name]-before[name]:+.4f} |")
    lines += ["", "## Record disposition", "",
              "| Table | Raw | Retained | Normalized | Deduplicated | Quarantined |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for table in TABLES:
        actions = report["cleaning"][table]["actions"]
        lines.append(f"| {table} | {report['counts']['before'][table]['total']} | "
                     f"{actions.get('keep', 0)} | "
                     f"{report['cleaning'][table]['reasons'].get('normalized', 0)} | "
                     f"{actions.get('deduplicate', 0)} | {actions.get('quarantine', 0)} |")
    lines += ["", "## Detected issues and evidence", ""]
    for table in TABLES:
        lines.append(f"### {table}")
        reasons = report["cleaning"][table]["reasons"]
        lines.append("; ".join(f"{key}: {value}" for key, value in sorted(reasons.items())) or "No rows needed disposition.")
        attempted = report["cleaning"][table]["repair_attempts"]
        accepted = report["cleaning"][table]["repair_reasons"]
        lines.append("Attempted unambiguous repairs: " +
                     ("; ".join(f"{key}: {value}" for key, value in sorted(attempted.items())) or "none"))
        lines.append("Repairs retained after conflict and reference checks: " +
                     ("; ".join(f"{key}: {value}" for key, value in sorted(accepted.items())) or "none"))
        for sample in report["cleaning"][table]["examples"][:5]:
            lines.append(f"- {sample['action']} / {sample['reason']}: {sample['raw']}")
        lines.append("")
    split = report["time_split"]
    lines += ["## Chronological boundaries for later iterations", "",
              f"T1: {split['T1']} ({split['T1_utc']})",
              f"T2: {split['T2']} ({split['T2_utc']})",
              f"Train / validation / test ratings: {split['counts']['train']} / "
              f"{split['counts']['validation']} / {split['counts']['test']}",
              f"Method: {split['method']}", "",
              "## Scoring scope and limitations", "",
              "- Scores use the same checks and UTC reference for both stages and range from 0 to 100.",
              "- Accurate is a rule-conformance proxy. Real-world truth of ratings, demographics, and metadata cannot be verified here.",
              "- Complete measures required fields; Unique discounts exact repeated rows.",
              "- Up-to-date is the fraction of valid ratings from the preceding 365 days. The historical dataset does not become recent through cleaning.",
              "- Consistent checks domains and rating foreign keys. The first, second, third, and fifth scores weight all three tables by row count; timeliness uses ratings only.",
              "- Deduplication and quarantine reduce the denominator and do not repair the source. Inspect the quarantined rows and Hadoop logs.",
              "- Future training inputs must use ratings at or before T1; validation/test information cannot enter training features.", ""]
    return "\n".join(lines)


def run_pipeline(progress: Callable[[str], None] | None = None) -> dict:
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(ZIP_PATH)
    task_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run = RUNS / task_id
    for folder in ("raw", "clean", "quarantine", "hadoop", "logs"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    state = {"task_id": task_id, "status": "running", "stage": "ingest",
             "started_at": utc_now(), "source_sha256": digest(ZIP_PATH)}
    atomic_json(run / "status.json", state)
    jobs = []

    def step(name: str) -> None:
        state["stage"] = name
        atomic_json(run / "status.json", state)
        if progress:
            progress(name)

    try:
        step("hadoop-runtime-check")
        check_hadoop_runtime()
        step("ingest")
        input_hashes = ingest(ZIP_PATH, run / "raw")
        reference = int(datetime.now(timezone.utc).timestamp())
        raw_users, raw_movies = run / "raw" / "users.dat", run / "raw" / "movies.dat"
        clean_users, clean_movies = run / "clean" / "users.dat", run / "clean" / "movies.dat"
        before = {}
        for table in TABLES:
            step(f"score-before-{table}")
            before[table] = score_table(run, "raw", table, raw_users, raw_movies, reference, jobs)
        cleaning = {}
        for table in TABLES:
            step(f"clean-{table}")
            cleaning[table] = clean_table(run, table, clean_users if table == "ratings" else None,
                                          clean_movies if table == "ratings" else None, jobs)
        after = {}
        for table in TABLES:
            step(f"score-after-{table}")
            after[table] = score_table(run, "clean", table, clean_users, clean_movies, reference, jobs)
        step("time-split-and-report")
        split = time_split(run / "clean" / "ratings.dat")
        data_version = hashlib.sha256("".join(cleaning[t]["clean_sha256"] for t in TABLES).encode()).hexdigest()
        report = {
            **state, "status": "completed", "stage": "completed", "finished_at": utc_now(),
            "input_file_sha256": input_hashes, "data_version": data_version,
            "rule_version": RULE_VERSION, "score_version": SCORE_VERSION,
            "hadoop_image": IMAGE, "hadoop_jobs": jobs,
            "reference_utc": datetime.fromtimestamp(reference, timezone.utc).isoformat(),
            "counts": {"before": before, "after": after},
            "scores": {"before": aggregate_scores(before), "after": aggregate_scores(after)},
            "cleaning": cleaning, "time_split": split,
            "paths": {"clean": str((run / "clean").relative_to(ROOT)),
                      "quarantine": str((run / "quarantine").relative_to(ROOT)),
                      "logs": str((run / "logs").relative_to(ROOT))},
        }
        atomic_json(run / "report.json", report)
        (run / "report.md").write_text(report_markdown(report), encoding="utf-8")
        atomic_json(run / "status.json", {key: state[key] for key in ("task_id", "started_at",
                    "source_sha256")} | {"status": "completed", "stage": "completed",
                                          "finished_at": report["finished_at"]})
        return report
    except Exception as exc:
        state.update({"status": "failed", "finished_at": utc_now(), "error": str(exc),
                      "hadoop_jobs": jobs})
        atomic_json(run / "status.json", state)
        raise
