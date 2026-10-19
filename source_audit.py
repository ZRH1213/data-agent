"""Read-only anomaly inventory for rule design; not the Hadoop quality report."""
import json
import zipfile
from collections import Counter

from pipeline import ZIP_PATH, TABLES


def audit() -> dict:
    findings = {}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for table in TABLES:
            expected = {"ratings": 4, "users": 5, "movies": 3}[table]
            shapes = Counter()
            unusual = []
            timestamps = Counter()
            scores = Counter()
            lines = 0
            with archive.open(f"ml-1m/{table}.dat") as source:
                for raw in source:
                    lines += 1
                    text = raw.decode("iso-8859-1").rstrip("\r\n")
                    parts = text.split("::")
                    shape = f"double_colon_{len(parts)}"
                    if len(parts) == 1:
                        shape = f"comma_{len(text.split(','))}" if "," in text else "other"
                    shapes[shape] += 1
                    if shape != f"double_colon_{expected}" and len(unusual) < 12:
                        unusual.append(text[:160])
                    if table == "ratings":
                        fields = parts if len(parts) == 4 else text.split(",")
                        if len(fields) == 4:
                            scores[fields[2].strip()] += 1
                            stamp = fields[3].strip()
                            timestamps[len(stamp)] += 1
            findings[table] = {
                "lines": lines, "shapes": dict(shapes), "unusual_samples": unusual,
                "timestamp_digit_lengths": dict(timestamps) if table == "ratings" else None,
                "rating_values": dict(scores) if table == "ratings" else None,
            }
    return findings


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
