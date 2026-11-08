"""Replay quantized candidate archives and report paired evidence-context intervals."""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1] / "results"
GRID = np.linspace(0, 1, 5)
METHODS = ("base", "half", "full", "raw", "constant", "oracle_grid", "point50")
REPLICATES = 5000


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def index(rows):
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("Duplicate record identifiers")
    return by_id


def validate_archive(path):
    protocol = read_json(path / "protocol.json")
    cases = read_rows(path / "cases.jsonl.gz")
    requests = read_rows(path / "requests.jsonl.gz")
    assert digest(cases) == protocol["cases_sha256"]
    assert digest(requests) == protocol["requests_sha256"]
    assert all(digest(req["prompt"]) == req["prompt_sha256"] for req in requests)
    assert len({req["prompt_sha256"] for req in requests}) == len(requests)
    cases_by_id = index(cases)
    requests_by_id = index(requests)
    predictions = {}
    for model in protocol["selector_models"]:
        rows = index(read_rows(path / f"predictions_{model}.jsonl.gz"))
        expected = {
            req["id"] for req in requests
            if req["kind"] == "select" or model == protocol["common_auditor"]
        }
        assert set(rows) == expected
        metadata = read_json(path / f"model_{model}.json")
        assert metadata["completed"] and metadata["total_requests"] == len(rows)
        for key, row in rows.items():
            req = requests_by_id[key]
            logits = np.asarray(row["logits"])
            perm = req["permutation"]
            assert row["prompt_sha256"] == req["prompt_sha256"]
            assert len(logits) == len(perm) and sorted(perm) == list(range(len(perm)))
            assert np.isfinite(logits).all()
            assert 0 <= row["candidate_mass"] <= 1 + 1e-8
            if row["selected_token"] in row["choice_token_ids"]:
                assert row["selected_token"] == row["choice_token_ids"][np.argmax(logits)]
        predictions[model] = rows
    return protocol, cases_by_id, requests, predictions


def fit_calibration(raw, truth, regularization):
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    features = np.column_stack([np.log(clipped / (1 - clipped)), np.ones(len(raw))])
    reference = np.array([1., 0.])

    def objective(weight):
        predicted = expit(features @ weight)
        residual = predicted - truth
        displacement = weight - reference
        loss = np.mean(residual ** 2) + regularization * (displacement @ displacement)
        derivative = 2 * residual * predicted * (1 - predicted)
        gradient = (features.T @ derivative) / len(truth) + 2 * regularization * displacement
        return loss, gradient

    fit = minimize(objective, reference, jac=True, method="L-BFGS-B",
                   bounds=[(0., 10.), (-10., 10.)],
                   options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000, "maxls": 80})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def calibration_data(cases, raw_reports):
    train, validation = [[case for case in cases.values() if case["split"] == split]
                         for split in ("train", "validation")]

    def data(split):
        return (np.array([np.mean(raw_reports[case["id"], "base"]) for case in split]),
                np.array([case["p"] for case in split]))

    return data(train), data(validation)


def select_calibration(train, validation, regularizations):
    train_raw, train_truth = train
    val_raw, val_truth = validation
    val_raw = np.clip(val_raw, 1e-6, 1 - 1e-6)
    val_logit = np.log(val_raw / (1 - val_raw))
    trials = []
    for regularization in regularizations:
        weight = fit_calibration(train_raw, train_truth, regularization)
        risk = np.mean((expit(weight[0] * val_logit + weight[1]) - val_truth) ** 2)
        trials.append((risk, regularization, weight))
    return min(trials, key=lambda item: item[0])


def verify_calibration(cases, raw_reports, protocol, calibration):
    train, validation = calibration_data(cases, raw_reports)
    regularizations = protocol["calibration_regularization_grid"]
    risk, regularization, weight = select_calibration(train, validation, regularizations)
    expected = calibration["selected"]
    assert regularization == expected["regularization"]
    assert np.allclose(weight, expected["weight"], rtol=0, atol=2e-7)
    assert abs(risk - expected["validation_regret"]) < 1e-8


def target(center, temperature):
    return softmax(-(GRID - center) ** 2 / temperature)


def candidate_metrics(prob, reports, truth):
    mean = float(prob @ reports)
    bias_squared = (mean - truth) ** 2
    variance = float(prob @ (reports - mean) ** 2)
    risk = float(prob @ (reports - truth) ** 2)
    assert abs(risk - bias_squared - variance) < 1e-12
    return risk, mean, bias_squared, variance


def analyse(dataset):
    path = ROOT / dataset
    protocol, cases, requests, predictions = validate_archive(path)
    calibrator = read_json(path / "auditor_calibration.json")
    assert not calibrator["test_truth_used_for_fit"]
    slope, intercept = calibrator["selected"]["weight"]
    training_mean = calibrator["training_mean"]
    if dataset == "scale":
        counts = {split: sum(c["split"] == split for c in cases.values())
                  for split in ("train", "validation", "test")}
        assert counts == {"train": 256, "validation": 64, "test": 128}
        assert len({(c["prior"], c["l1"], c["l0"]) for c in cases.values()}) == len(cases)
        for case in cases.values():
            prior, l1, l0 = (case[key] for key in ("prior", "l1", "l0"))
            numerator = prior * l1
            posterior = numerator / (numerator + (100 - prior) * l0)
            f1, f0 = case["f1"], case["f0"]
            posterior_new = posterior * f1 / (posterior * f1 + (1 - posterior) * f0)
            assert abs(posterior - case["p"]) < 1e-12
            assert abs(posterior_new - case["p_new"]) < 1e-12
        assert abs(training_mean - np.mean([c["p"] for c in cases.values()
                                            if c["split"] == "train"])) < 1e-12
    else:
        source = read_json(ROOT / "scale" / "auditor_calibration.json")
        assert calibrator["selected"] == source["selected"]
        assert training_mean == source["training_mean"]
        for case in cases.values():
            assert case["p"] == float(case["proposed_answer"] == case["gold_answer"])

    auditor = predictions[protocol["common_auditor"]]
    raw_reports = defaultdict(list)
    for req in requests:
        if req["kind"] == "audit":
            scores = softmax(auditor[req["id"]]["logits"])
            raw_reports[req["case_id"], req["condition"]].append(
                float(scores @ GRID[req["permutation"]])
            )
    assert all(len(values) == 2 for values in raw_reports.values())
    if dataset == "scale":
        verify_calibration(cases, raw_reports, protocol, calibrator)

    temperature = protocol["target_temperature"]
    constant = target(training_mean, temperature)
    point50 = np.asarray([0., 0., 1., 0., 0.])
    per_prompt = []
    for model, pred in predictions.items():
        for req in requests:
            if req["kind"] != "select":
                continue
            cid, condition = req["case_id"], req["condition"]
            phase = "new_evidence" if condition == "new_evidence" else "base"
            truth = cases[cid]["p_new" if phase == "new_evidence" else "p"]
            raw = np.mean(raw_reports[cid, phase])
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            center = expit(slope * np.log(clipped / (1 - clipped)) + intercept)
            full = target(center, temperature)
            probs = softmax(pred[req["id"]]["logits"])
            classes = np.asarray(req["permutation"]) // 2
            base = np.bincount(classes, weights=probs, minlength=len(GRID))
            policies = [base, (base + full) / 2, full, target(raw, temperature),
                        constant, target(truth, temperature), point50]
            for method, policy in zip(METHODS, policies):
                risk, mean, bias_squared, variance = candidate_metrics(policy, GRID, truth)
                per_prompt.append(dict(dataset=dataset, model=model, context=cid,
                    condition=condition, stance=req["stance"], method=method, truth=truth,
                    risk=risk, mean_report=mean, squared_bias=bias_squared, variance=variance))
    return per_prompt, sum(map(len, predictions.values())), (cases, raw_reports, protocol)


def paired_results(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["model"], row["condition"], row["method"])
        groups[key][row["context"], row["stance"]] = row
    rng = np.random.default_rng(20261107)
    output = []

    def add(dataset, model, condition, metric, method, comparator, differences):
        contexts = sorted(differences)
        assert all(len(differences[c]) in {1, 2} for c in contexts)
        delta = np.asarray([np.mean(differences[c]) for c in contexts])
        indices = rng.integers(len(delta), size=(REPLICATES, len(delta)))
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        output.append(dict(dataset=dataset, model=model, condition=condition,
            metric=metric, method=method, comparator=comparator,
            mean=float(delta.mean()), ci_low=float(low), ci_high=float(high),
            contexts=len(contexts), replicates=REPLICATES))

    for (dataset, model, condition, method), current in sorted(groups.items()):
        # Full, raw and constant share their report policies across selectors.
        if method not in {"half", "full", "raw"} or (method != "half" and model != "qwen15"):
            continue
        reference = "base" if method == "half" else "constant"
        comparison = groups[dataset, model, condition, reference]
        for metric in ("risk", "squared_bias", "variance"):
            differences = defaultdict(list)
            for key, row in current.items():
                differences[key[0]].append(row[metric] - comparison[key][metric])
            add(dataset, model, condition, metric, method, reference, differences)
        if method == "full":
            point = groups[dataset, model, condition, "point50"]
            differences = defaultdict(list)
            for key, row in current.items():
                differences[key[0]].append(row["risk"] - point[key]["risk"])
            add(dataset, model, condition, "risk", method, "point50", differences)

    for (dataset, model, condition, method), current in sorted(groups.items()):
        if condition != "new_evidence":
            continue
        if model != "qwen15" and method in {"full", "raw", "constant", "oracle_grid", "point50"}:
            continue
        before = groups[dataset, model, "base", method]
        comparisons = defaultdict(list)
        response = defaultdict(list)
        for key, row in current.items():
            shift = row["mean_report"] - before[key]["mean_report"]
            truth_shift = row["truth"] - before[key]["truth"]
            response[key[0]].append(abs(shift))
            # Zero update is the evidence-blind baseline; positive is worse.
            comparisons[key[0]].append(abs(shift - truth_shift) - abs(truth_shift))
        add(dataset, model, condition, "absolute_update", method, "zero", response)
        add(dataset, model, condition, "update_mae", method, "zero", comparisons)
    for (dataset, model, condition, method), current in sorted(groups.items()):
        if condition not in {"base", "new_evidence"} or dataset != "scale":
            continue
        if model != "qwen15" and method not in {"base", "half"}:
            continue
        by_context = defaultdict(dict)
        for (cid, stance), row in current.items():
            by_context[cid][stance] = row["mean_report"]
        signed = {}
        for cid, values in by_context.items():
            assert set(values) == {.1, .9}
            signed[cid] = [values[.9] - values[.1]]
        absolute = {cid: [abs(values[0])] for cid, values in signed.items()}
        add(dataset, model, condition, "stance_gap", method, "zero", absolute)
        add(dataset, model, condition, "stance_signed", method, "zero", signed)
    return output


def refit_sensitivity(bayes, external):
    """Resample calibration splits and test contexts without using test labels for fit."""
    cases, raw, protocol = bayes
    train, validation = calibration_data(cases, raw)
    rng = np.random.default_rng(202611071)
    draws = defaultdict(list)

    def test_arrays(data, phase):
        ds_cases, ds_raw, _ = data
        test = [c for c in ds_cases.values() if c["split"] == "test"]
        phase_key = "p_new" if phase == "new_evidence" else "p"
        values = np.asarray([np.mean(ds_raw[c["id"], phase]) for c in test])
        return values, np.asarray([c[phase_key] for c in test])

    bayes_raw, bayes_truth = test_arrays(bayes, "base")
    added_raw, added_truth = test_arrays(bayes, "new_evidence")
    external_raw, external_truth = test_arrays(external, "base")
    temperature = protocol["target_temperature"]

    def grid(center):
        center = np.atleast_1d(center)
        return softmax(-(GRID - center[:, None]) ** 2 / temperature, axis=1)

    for _ in range(750):
        ti = rng.integers(len(train[0]), size=len(train[0]))
        vi = rng.integers(len(validation[0]), size=len(validation[0]))
        _, _, weight = select_calibration(
            tuple(values[ti] for values in train),
            tuple(values[vi] for values in validation),
            protocol["calibration_regularization_grid"],
        )

        def calibrated(values):
            values = np.clip(values, 1e-6, 1 - 1e-6)
            return grid(expit(weight[0] * np.log(values / (1 - values)) + weight[1]))

        constant = grid(train[1][ti].mean())[0]
        bayes_policy = calibrated(bayes_raw)
        bi = rng.integers(len(bayes_truth), size=len(bayes_truth))
        ei = rng.integers(len(external_truth), size=len(external_truth))
        for name, policy, truth, ix in (
            ("scale", bayes_policy, bayes_truth, bi),
            ("external_math", calibrated(external_raw), external_truth, ei),
        ):
            regret_delta = np.sum((policy - constant) * (GRID - truth[:, None]) ** 2, axis=1)
            draws[name, "risk_minus_constant"].append(regret_delta[ix].mean())
            risk = np.sum(policy * (GRID - truth[:, None]) ** 2, axis=1)
            draws[name, "risk_minus_point50"].append((risk - (.5 - truth) ** 2)[ix].mean())
        shift = (calibrated(added_raw) - bayes_policy) @ GRID
        truth_shift = added_truth - bayes_truth
        update_error = abs(shift - truth_shift) - abs(truth_shift)
        draws["scale", "update_mae_minus_zero"].append(update_error[bi].mean())

    rows = []
    for (dataset, metric), values in draws.items():
        low, high = np.quantile(values, [.025, .975])
        rows.append(dict(dataset=dataset, metric=metric, ci_low=float(low),
                         ci_high=float(high), replicates=len(values)))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_paper_table(comparisons):
    names = {
        "base": "Qwen 1.5B base", "half": "Qwen 1.5B half repair",
        "full": "Calibrated full repair", "raw": "Raw auditor",
        "constant": "Zero update", "oracle_grid": "Oracle grid",
    }
    entries = {(r["method"], r["metric"]): r for r in comparisons
               if r["model"] == "qwen15" and r["dataset"] == "scale"
               and ((r["condition"] == "new_evidence" and r["metric"] in {"absolute_update", "update_mae"})
                    or (r["condition"] == "base" and r["metric"] == "stance_gap"))}
    eol = " " + chr(92) * 2
    latex = [
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Policy & Stance gap & Absolute update & Error difference & 95\% interval" + eol,
        r"\midrule",
    ]
    for method, name in names.items():
        gap = entries[method, "stance_gap"]
        update = entries[method, "absolute_update"]
        error = entries[method, "update_mae"]
        latex.append(f"{name} & {gap['mean']:.5f} & {update['mean']:.5f} & ${error['mean']:.5f}$ & "
                     f"$[{error['ci_low']:.5f}, {error['ci_high']:.5f}]$" + eol)
    latex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT.parent / "paper" / "table_update_replay.tex").write_text("\n".join(latex) + "\n")


def main():
    bayes, bayes_count, bayes_data = analyse("scale")
    external, external_count, external_data = analyse("external_math")
    rows = bayes + external
    write_csv(ROOT / "calibration_refit_intervals.csv",
              refit_sensitivity(bayes_data, external_data))
    comparisons = paired_results(rows)
    write_csv(ROOT / "evidence_replay_per_prompt.csv", rows)
    write_csv(ROOT / "evidence_replay_intervals.csv", comparisons)
    write_paper_table(comparisons)
    summary = dict(archived_predictions={"scale": bayes_count, "external_math": external_count},
        fresh_inference=False, archived_calibrator_fit_verified=True, calibrator_refit_sensitivity_replicates=750,
        bootstrap="Paired evidence-context resampling, fixed model, calibrator and design",
        exploratory=True, per_prompt_metrics=len(rows), comparisons=len(comparisons))
    (ROOT / "evidence_replay_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
