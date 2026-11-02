"""Exploratory auditor diagnostics; replay scores, never claim new inference.

The formula solver reads public evidence text only. It is a task-specific
competitor, not an oracle label lookup or a general language understanding model.
"""
from pathlib import Path
import argparse, csv, json, re
import numpy as np
from scale_study import ROOT, QV, TEMP, digest, load_rows, softmax, write_csv, write_json


def solve_public_evidence(text):
    patterns = [r'P\(H\)=(\d+)%. P\(E\|H\)=(\d+)%. P\(E\|not H\)=(\d+)%',
                r'Among all cases, (\d+)% have H. The chance of seeing E is (\d+)% for cases with H and (\d+)% for cases without H']
    match = next((m for pat in patterns if (m := re.search(pat, text))), None)
    if match is None:
        raise ValueError('Unsupported public evidence syntax')
    prior, yes, no = map(int, match.groups())
    if not all(0 < value < 100 for value in (prior, yes, no)):
        raise ValueError('Evidence probabilities must be strictly between 0 and 100')
    odds = prior / (100 - prior) * yes / no
    extra = re.search(r'P\(F\|H\)=(\d+)%. P\(F\|not H\)=(\d+)%', text)
    if extra:
        yes2, no2 = map(int, extra.groups())
        if not (0 < yes2 < 100 and 0 < no2 < 100):
            raise ValueError('Invalid additional evidence')
        if 'conditionally independent' not in text:
            raise ValueError('Cannot multiply likelihood ratios without independence')
        odds *= yes2 / no2
    return odds / (1 + odds)


def paired_interval(delta, rng):
    indices = rng.integers(0, len(delta), (5000, len(delta)))
    return np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()


def run(out):
    source = ROOT / 'results/scale'
    cases = load_rows(source / 'cases.jsonl.gz')
    requests = load_rows(source / 'requests.jsonl.gz')
    protocol = json.loads((source / 'protocol.json').read_text())
    if digest(cases) != protocol['cases_sha256'] or digest(requests) != protocol['requests_sha256']:
        raise ValueError('Source protocol changed')
    # Build the method output before constructing any map containing labels.
    solved = {r['id']: solve_public_evidence(r['prompt']) for r in requests}
    labels = {c['id']: c for c in cases}
    max_error = max(abs(solved[r['id']] - labels[r['case_id']][
        'p_new' if r['condition'] == 'new_evidence' else 'p']) for r in requests)
    if max_error > 1e-12:
        raise ValueError('Public-evidence solver disagrees with archived evaluation labels')
    # Mutation check: the predictor has only a text argument, not a case object.
    mutated = [{**c, 'p': 1-c['p'], 'p_new': 1-c['p_new']} for c in cases]
    if not any(a['p'] != b['p'] for a, b in zip(cases, mutated)):
        raise ValueError('Label mutation did not change the evaluation fixture')
    if solved != {r['id']: solve_public_evidence(r['prompt']) for r in requests}:
        raise ValueError('Predictions changed under unrelated label mutation')
    audit = list(csv.DictReader((source / 'auditor_predictions.csv').open()))
    test = {c['id']: c for c in cases if c['split'] == 'test'}
    rows = []
    for r in audit:
        if r['split'] != 'test':
            continue
        case = test[r['case_id']]
        variant = r['condition']
        request = next(x for x in requests if x['id'] == f"audit:{case['id']}:{variant}:0")
        solver = solved[request['id']]
        for method, estimate in [('Raw-LM-auditor', float(r['raw_mean'])),
                                 ('Calibrated-LM-auditor', float(r['calibrated_mean'])),
                                 ('Public-evidence-formula', solver)]:
            tau = softmax(-(QV-estimate)**2/TEMP)
            mean = float(tau @ QV)
            variance = float(tau @ (QV-mean)**2)
            truth = float(r['p'])
            point_error = (estimate-truth)**2
            grid_bias = (mean-estimate)**2
            interaction = 2*(mean-estimate)*(estimate-truth)
            risk = float(tau @ (QV-truth)**2)
            if abs(risk-(variance+point_error+grid_bias+interaction)) > 1e-12:
                raise ValueError('Risk decomposition failed')
            rows.append(dict(case_id=case['id'], condition=variant, method=method,
                estimate=estimate, truth=truth, mean_report=mean, point_regret=point_error,
                full_report_regret=risk, report_variance=variance,
                grid_bias_squared=grid_bias, grid_audit_interaction=interaction,
                nearest_report_regret=float((QV[np.argmin((QV-estimate)**2)]-truth)**2)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_case.csv', rows)
    rng = np.random.default_rng(2026110241)
    summary = []
    for condition in ['base', 'new_evidence']:
        for method in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['condition'] == condition and r['method'] == method]
            summary.append(dict(condition=condition, method=method, contexts=len(group), **{
                name: float(np.mean([r[name] for r in group])) for name in
                ['point_regret', 'full_report_regret', 'report_variance', 'grid_bias_squared',
                 'grid_audit_interaction', 'nearest_report_regret']}))
    write_csv(out / 'summary.csv', summary)
    updates = []
    for method in dict.fromkeys(r['method'] for r in rows):
        by = {(r['case_id'], r['condition']): r for r in rows if r['method'] == method}
        truth, changes, raw_changes = [], [], []
        for cid, c in test.items():
            before, after = by[cid, 'base'], by[cid, 'new_evidence']
            truth.append(c['p_new']-c['p'])
            changes.append(after['mean_report']-before['mean_report'])
            raw_changes.append(after['estimate']-before['estimate'])
        truth, changes, raw_changes = map(np.asarray, [truth, changes, raw_changes])
        updates.append(dict(method=method, contexts=len(test),
            mean_absolute_update=float(np.mean(abs(changes))),
            evidence_update_magnitude_ratio=float(np.mean(abs(changes))/np.mean(abs(truth))),
            audit_estimate_update_magnitude_ratio=float(np.mean(abs(raw_changes))/np.mean(abs(truth))),
            correct_direction_rate=float(np.mean(changes*truth>1e-12)),
            update_squared_error=float(np.mean((changes-truth)**2))))
    write_csv(out / 'updates.csv', updates)
    paired = []
    for condition in ['base', 'new_evidence']:
        by = {(r['case_id'], r['method']): r for r in rows if r['condition'] == condition}
        for metric in ['point_regret', 'full_report_regret']:
            delta = np.array([by[cid, 'Public-evidence-formula'][metric]-
                              by[cid, 'Calibrated-LM-auditor'][metric] for cid in test])
            lo, hi = paired_interval(delta, rng)
            paired.append(dict(condition=condition, metric=metric,
                comparison='formula minus calibrated LM', mean_difference=float(delta.mean()),
                ci95_low=lo, ci95_high=hi, contexts=len(test), bootstrap_replicates=5000))
    write_csv(out / 'paired_intervals.csv', paired)
    write_json(out / 'status.json', dict(completed=True, fresh_model_inference=False,
        exploratory=True, independent_new_test_cases=0,
        source_protocol_sha256=digest(protocol), source_requests_sha256=digest(requests),
        public_evidence_solver_max_label_error=max_error, label_mutation_check='passed',
        risk_decomposition_check='passed', bootstrap_scope='paired test contexts; archived auditor fit fixed',
        limitations=['Task-specific grammar and known Bayesian assumptions',
                    'Solver is not a general free-text verifier',
                    'Post-hoc diagnostic on already inspected test cases',
                    'Full-repair risk is identical for audit-only and all backbones']))
    print(json.dumps(dict(summary=summary, updates=updates, paired=paired), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/auditor_diagnostics')
    run(parser.parse_args().out)
