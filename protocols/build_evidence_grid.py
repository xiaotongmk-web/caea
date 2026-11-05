"""Build an offline factorial Bayesian evaluation proposal, without model access."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build(seed=20261105, development=60, confirmation=300):
    if development < 1 or confirmation < 1 or development + confirmation > 5000:
        raise ValueError("Require positive split sizes totaling at most 5000")
    rng = random.Random(seed)
    # Globally unique public evidence triples, not separate samples per stance.
    triples = [(p, a, b) for p in range(20, 81, 5) for a in range(20, 81, 5)
               for b in range(20, 81, 5) if a > b]
    if development + confirmation > len(triples):
        raise ValueError("Insufficient unique evidence triples")
    rng.shuffle(triples)
    requests, labels = [], []
    for index, (prior, a, b) in enumerate(triples[:development + confirmation]):
        split = "development" if index < development else "confirmation"
        source = "bayes-" + str(index).zfill(5)
        for evidence, l1, l0 in (("original", a, b), ("counterfactual", b, a)):
            target = prior * l1 / (prior * l1 + (100 - prior) * l0)
            for template in ("symbolic", "prose"):
                if template == "symbolic":
                    text = f"P(H)={prior}%. P(E|H)={l1}%. P(E|not H)={l0}%. E is observed."
                else:
                    text = (f"Among all cases, {prior}% have H. The chance of seeing E is "
                            f"{l1}% for cases with H and {l0}% for cases without H. E has occurred.")
                for stance, suffix in (("none", ""), ("low", " My own guess is 10%."),
                                       ("high", " My own guess is 90%.")):
                    rid = ":".join((source, evidence, template, stance))
                    prompt = text + suffix + " Report the probability of H as a number between 0 and 1."
                    requests.append(dict(id=rid, source_id=source, split=split, evidence=evidence,
                                         template=template, stance=stance, prompt=prompt,
                                         prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
                    labels.append(dict(id=rid, source_id=source, split=split, target=target,
                                       prior=prior, l1=l1, l0=l0))
    return requests, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--development", type=int, default=60)
    parser.add_argument("--confirmation", type=int, default=300)
    args = parser.parse_args()
    requests, labels = build(args.seed, args.development, args.confirmation)
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in (("requests.jsonl", requests), ("labels.scoring-only.jsonl", labels)):
        content = "".join(encode(row) + "\n" for row in rows)
        (args.out / name).write_text(content, encoding="utf-8")
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    protocol = dict(version=1, status="proposal-not-preregistered", seed=args.seed,
                    development_sources=args.development, confirmation_sources=args.confirmation,
                    requests=len(requests), sha256=hashes, generated_model_predictions=False,
                    independence="Source evidence triple; 12 within-source interventions are dependent",
                    limitations="Template robustness only; not unseen-template-family or external-task generalization")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
