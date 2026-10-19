"""Hadoop Streaming jobs for MovieLens data governance."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

AGES = {"1", "18", "25", "35", "45", "50", "56"}
GENRES = {"Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"}
EARLIEST = 946684800
LATEST = 1046476799


def parse_record(table, raw):
    """Return normalized row, logical key, and error code."""
    count = {"ratings": 4, "users": 5, "movies": 3}[table]
    fields = raw.rstrip("\r\n").split("::")
    if len(fields) != count:
        return None, None, "field_count"
    if any(not x.strip() for x in fields):
        return None, None, "required_missing"
    f = [x.strip() for x in fields]
    try:
        if table == "ratings":
            user, movie, rating, stamp = map(int, f)
            if user <= 0 or movie <= 0 or rating not in range(1, 6):
                return None, None, "rating_domain"
            if not EARLIEST <= stamp <= LATEST:
                return None, None, "timestamp_range"
            return f"{user}::{movie}::{rating}::{stamp}", f"{user}:{movie}:{stamp}", None
        if table == "users":
            user, occupation = int(f[0]), int(f[3])
            if user <= 0 or f[1] not in {"M", "F"} or f[2] not in AGES:
                return None, None, "user_domain"
            if occupation not in range(21):
                return None, None, "user_domain"
            if not re.fullmatch(r"[A-Za-z0-9-]{1,12}", f[4]):
                return None, None, "zip_format"
            return f"{user}::{f[1]}::{f[2]}::{occupation}::{f[4]}", str(user), None
        movie = int(f[0])
        if movie <= 0:
            return None, None, "movie_id"
        title = re.sub(r"\s+", " ", f[1]).strip()
        genres = list(dict.fromkeys(x.strip() for x in f[2].split("|")))
        if any(x not in GENRES for x in genres):
            return None, None, "genre_domain"
        return f"{movie}::{title}::{'|'.join(genres)}", str(movie), None
    except ValueError:
        return None, None, "number_format"


def repair_record(table, raw):
    """Apply only reversible or unambiguous repairs before strict validation."""
    value = raw.rstrip("\r\n")
    repairs = []
    if value.endswith("::EXTRA_FIELD"):
        value = value.removesuffix("::EXTRA_FIELD")
        repairs.append("drop_literal_extra_field")
    if table == "ratings" and "::" not in value and len(value.split(",")) == 4:
        value = "::".join(value.split(","))
        repairs.append("comma_delimiter")
    elif table == "users" and "::" not in value and len(value.split(":")) == 5:
        value = "::".join(value.split(":"))
        repairs.append("single_colon_delimiter")
    elif table == "movies" and "::" not in value and "|" in value:
        pieces = value.split("|", 2)
        if len(pieces) == 3 and pieces[0].isdigit():
            value = "::".join(pieces)
            repairs.append("pipe_delimiter")
    fields = value.split("::")
    if table == "ratings" and len(fields) == 4:
        if fields[2].strip().lower() == "five":
            fields[2] = "5"
            repairs.append("english_five")
        stamp = fields[3].strip()
        if stamp.isdigit() and len(stamp) in (12, 13):
            seconds = int(stamp) // 1000
            if EARLIEST <= seconds <= LATEST:
                fields[3] = str(seconds)
                repairs.append("milliseconds_to_seconds")
        value = "::".join(fields)
    row, key, error = parse_record(table, value)
    if row and row != value:
        repairs.append("whitespace_or_genre_normalization")
    return row, key, error, repairs


def parent_ids(path):
    if not path:
        return set()
    table = Path(path).stem
    if table not in {"users", "movies"}:
        raise ValueError(f"Expected users.dat or movies.dat, got {path}")
    ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            _row, key, error = parse_record(table, line)
            if not error:
                ids.add(key)
    return ids


def clean_map(table, users, movies):
    for source in sys.stdin:
        raw = source.rstrip("\r\n")
        row, key, error, repairs = repair_record(table, raw)
        if not error and table == "ratings":
            user, movie = row.split("::", 2)[:2]
            if user not in users or movie not in movies:
                error = "orphan_reference"
        item = {"raw": raw, "row": row, "error": error, "repairs": repairs}
        group = key if key and not error else "bad:" + hashlib.sha256(raw.encode()).hexdigest()
        print(group + "\t" + json.dumps(item, ensure_ascii=False))


def clean_reduce():
    current, rows = None, []

    def flush():
        if not rows:
            return
        distinct = {item["row"] for item in rows if not item["error"]}
        # Hadoop does not promise an order among identical shuffle keys.
        # Prefer an already-canonical source row so reruns make the same choice.
        ordered = sorted(rows, key=lambda item: (item["raw"] != item["row"], item["raw"]))
        for index, item in enumerate(ordered):
            if item["error"]:
                action, reason = "quarantine", item["error"]
            elif len(distinct) > 1:
                action, reason = "quarantine", "key_conflict"
            elif index:
                action, reason = "deduplicate", "same_logical_key"
            else:
                action, reason = "keep", "normalized" if item["raw"] != item["row"] else None
            print(json.dumps({"action": action, "reason": reason, "raw": item["raw"],
                              "row": item["row"] if action == "keep" else None,
                              "repairs": item["repairs"]},
                             ensure_ascii=False))

    for source in sys.stdin:
        key, payload = source.rstrip("\n").split("\t", 1)
        if current is not None and current != key:
            flush()
            rows = []
        current = key
        rows.append(json.loads(payload))
    flush()


def score_map(table, users, movies, reference):
    metrics = Counter()
    for source in sys.stdin:
        raw = source.rstrip("\r\n")
        fields = raw.split("::")
        row, _key, error = parse_record(table, raw)
        metrics["total"] += 1
        if len(fields) == {"ratings": 4, "users": 5, "movies": 3}[table] and all(x.strip() for x in fields):
            metrics["complete"] += 1
        if not error:
            metrics["accurate_proxy"] += 1
            consistent = True
            if table == "ratings":
                user, movie, _rating, stamp = row.split("::")
                consistent = user in users and movie in movies
                if reference - int(stamp) <= 365 * 86400 and int(stamp) <= reference:
                    metrics["fresh"] += 1
            if consistent:
                metrics["consistent"] += 1
        digest = hashlib.sha256(raw.encode()).hexdigest()
        print(f"duplicate:{digest}\t1")
    for name, count in metrics.items():
        print(f"metric:{name}\t{count}")


def score_reduce():
    current, total = None, 0

    def flush():
        if current is None:
            return
        if current.startswith("duplicate:"):
            if total > 1:
                print(f"duplicate_excess\t{total - 1}")
        else:
            print(f"{current.removeprefix('metric:')}\t{total}")

    for source in sys.stdin:
        key, value = source.rstrip("\n").split("\t", 1)
        if current is not None and key != current:
            flush()
            total = 0
        current = key
        total += int(value)
    flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["clean-map", "clean-reduce", "score-map", "score-reduce"])
    parser.add_argument("--table", choices=["ratings", "users", "movies"])
    parser.add_argument("--users")
    parser.add_argument("--movies")
    parser.add_argument("--reference", type=int)
    args = parser.parse_args()
    if args.mode == "clean-map":
        clean_map(args.table, parent_ids(args.users), parent_ids(args.movies))
    elif args.mode == "clean-reduce":
        clean_reduce()
    elif args.mode == "score-map":
        score_map(args.table, parent_ids(args.users), parent_ids(args.movies), args.reference)
    else:
        score_reduce()


if __name__ == "__main__":
    main()
