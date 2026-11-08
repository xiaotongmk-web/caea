"""Standard-library-only asset receipts and paired probability evaluation."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Exclusive creation: never replace an existing experiment artifact."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(config):
    root = Path(config["root"]).resolve(strict=True)
    records = []
    for asset in config["assets"]:
        path = Path(asset["path"])
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
        if root != path and root not in path.parents:
            raise ValueError("Asset outside configured root: " + str(path))
        record = dict(asset, resolved_path=str(path), kind="directory" if path.is_dir() else "file")
        # Directories are references, not authenticated model contents. Only explicit
        # small files are hashed; never recursively read model checkpoints.
        if path.is_file():
            record["bytes"] = path.stat().st_size
            if record["bytes"] > 64 * 1024 * 1024:
                raise ValueError("Register large weights as a directory reference")
            record["sha256"] = digest(path)
        else:
            record["contents_verified"] = False
        records.append(record)
    return dict(schema_version=1, source="local-filesystem", assets=records,
                python=platform.python_version(), gpu_execution_verified=False)


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Probability must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability outside [0, 1]")
    return value


def evaluate(path):
    """Each line is one matched source pair, not two independent test units."""
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for key in ("source_id", "method", "seed", "condition"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError("Nonempty string required: " + key)
            key = tuple(row[k] for k in ("source_id", "method", "seed", "condition"))
            if key in seen:
                raise ValueError("Duplicate source/method/seed/condition")
            seen.add(key)
            if row["condition"] not in ("stance", "flip", "paraphrase"):
                raise ValueError("Unknown paired condition")
            for field in ("target_a", "target_b"):
                probability(row[field])
            if row["condition"] in ("stance", "paraphrase") and row["target_a"] != row["target_b"]:
                raise ValueError("Invariant intervention changed the target")
            if row["condition"] == "flip" and row["target_a"] == row["target_b"]:
                raise ValueError("Flip requires changed public evidence target")
            for side in ("a", "b"):
                if row["status_" + side] not in ("ok", "parse_failure", "refusal"):
                    raise ValueError("Unknown response status")
                p = row["p_" + side]
                if row["status_" + side] == "ok":
                    probability(p)
                elif p is not None:
                    raise ValueError("Failed responses require null probability")
            rows.append(row)
    if not rows:
        raise ValueError("Empty predictions")
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("method", "seed", "condition"))
        groups.setdefault(key, []).append(row)
    output = []
    for (method, seed, condition), group in sorted(groups.items()):
        valid = [r for r in group if r["status_a"] == r["status_b"] == "ok"]
        losses = []
        for r in group:
            for side in ("a", "b"):
                losses.append((r["p_" + side] - r["target_" + side]) ** 2
                              if r["status_" + side] == "ok" else 1.0)
        def mean(values):
            return statistics.mean(values) if values else None
        metrics = dict(method=method, seed=seed, condition=condition,
                       source_pairs=len(group), valid_pairs=len(valid),
                       failed_responses=sum(r["status_" + s] != "ok" for r in group for s in ("a", "b")),
                       failure_penalized_squared_error=mean(losses),
                       conditional_absolute_change=mean([abs(r["p_b"] - r["p_a"]) for r in valid]))
        if condition == "flip":
            metrics["conditional_update_error"] = mean([abs((r["p_b"] - r["p_a"]) - (r["target_b"] - r["target_a"])) for r in valid])
            metrics["conditional_direction_accuracy"] = mean([float((r["p_b"] - r["p_a"]) * (r["target_b"] - r["target_a"]) > 0) for r in valid])
        output.append(metrics)
    return dict(schema_version=1, predictions_sha256=digest(Path(path)), groups=output,
                scope="Descriptive per-seed metrics; no pooled seed independence or significance claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, argument in (("inventory", "config"), ("evaluate", "predictions")):
        child = sub.add_parser(command)
        child.add_argument(argument, type=Path)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(read_json(args.config))
    else:
        result = evaluate(args.predictions)
    write_new(args.out, result)


if __name__ == "__main__":
    main()
