"""Publish the project-labeled judge corpus, results, and immutable run manifest."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "Crusadersk/quantsafe-judge-benchmark"
CORPUS_PATH = ROOT / "substrate" / "judge_corpus.json"
RESULTS_PATH = ROOT / "substrate" / "judge_results.json"


def _dataset_card(results: dict, run_name: str) -> str:
    agreement = results["agreement"]
    uncertainty = results["statistical_uncertainty"]
    selective = results["gold_validation"]["selective_consensus"]
    reports = {report["model"]: report for report in results["judges"]}
    qwen = reports["Qwen/Qwen3Guard-Gen-0.6B"]["metrics"]
    granite = reports["ibm-granite/granite-guardian-3.3-8b"]["metrics"]
    nemotron = reports[
        "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"
    ]["metrics"]
    kappa_ci = uncertainty["kappa"]
    top_two = uncertainty["top_two_accuracy"]

    return f"""---
license: apache-2.0
task_categories:
- text-classification
language:
- en
tags:
- safety
- refusal
- llm-safety
- guardrails
- judge-agreement
- quantization
- nemotron
- granite-guardian
- qwen3guard
- build-small-hackathon
pretty_name: QuantSafe Refusal-Safety Judge-Agreement Benchmark
size_categories:
- n<1K
configs:
- config_name: default
  data_files: judge_corpus.jsonl
---

# QuantSafe Refusal-Safety Judge-Agreement Benchmark

A focused, **project-authored and project-labeled** 40-item corpus for measuring
how three open guard-model families label `(prompt, response)` pairs as
`safe / unsafe`. It supports the Judge Agreement tab in
[QuantSafe Certifier](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier)
and the broader methodology in
[*Quality Is Not a Safety Proxy Under Quantization*](https://arxiv.org/abs/2606.10154).

## Files

- `judge_corpus.jsonl`: 12 clear-safe, 12 clear-unsafe, and 16 intentionally
  borderline items with project labels.
- `judge_verdicts.json`: promoted aggregate results and provenance.
- `runs/{run_name}`: immutable run manifest with code/corpus/model revisions,
  generation settings, reported precision, elapsed time, verdict digest, and
  per-output SHA-256 hashes.

## Results

| Guard model | Parameters | Project-label accuracy | Macro F1 |
|---|---:|---:|---:|
| NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 8.03B | {nemotron['accuracy']:.1%} | {nemotron['macro_f1']:.3f} |
| IBM Granite Guardian 3.3 8B | 8.17B | {granite['accuracy']:.1%} | {granite['macro_f1']:.3f} |
| Qwen3Guard-Gen-0.6B | 0.75B | {qwen['accuracy']:.1%} | {qwen['macro_f1']:.3f} |

- Fleiss' kappa: **{agreement['kappa']:.3f}**
- Zone-stratified bootstrap 95% CI:
  **{kappa_ci['ci_low']:.3f}–{kappa_ci['ci_high']:.3f}**
  ({kappa_ci['n_resamples']:,} deterministic resamples)
- Unanimous non-unclear coverage: **{selective['coverage']:.1%}**
- Accuracy on covered items: **{selective['accuracy']:.1%}**
- Nemotron vs Granite exact paired McNemar:
  **p={top_two['two_sided_p_value']:.1f}**

The Nemotron guard has the highest point estimate by one item. The paired test
does not statistically separate it from Granite on this small corpus. The
kappa interval also crosses the predeclared 0.70 `RELIABLE` threshold, so the
band is a point-estimate classification rather than a certainty claim.

## Annotation and lineage

The 40 rows were authored for this project to exercise three zones:
straightforward benign responses, straightforward unsafe assistance, and
borderline fictional/transformative/contextual responses. The `expected` field
is a single-author project label used for an engineering agreement probe.

There was no independent annotation panel, annotator blinding, or adjudication.
All 16 borderline rows currently carry the project label `safe`; this makes the
corpus useful for surfacing conservative false positives, but it is also a
known design limitation. The rows are not presented as samples copied from
WildGuardMix or XSTest.

WildGuardMix and XSTest are used elsewhere in QuantSafe to train and evaluate
the separate ModernBERT refusal cross-check. They are not the source lineage of
this 40-row judge corpus.

## Reproduction

The promoted run used pinned model revisions and the authenticated Modal
`/judge` backend. Nemotron reported native BF16; Granite and Qwen reported
FP16. Raw prompt/response rows are public in `judge_corpus.jsonl`. Raw model
completions are not republished; the immutable manifest records their byte
lengths and SHA-256 hashes.

The benchmark is cohort-level support evidence. It is not a per-configuration
safety judgment, proof of model safety, or large-scale leaderboard.

## Citation

```bibtex
@misc{{kadadekar2026rtsi,
  title         = {{Quality Is Not a Safety Proxy Under Quantization}},
  author        = {{Kadadekar, Sahil}},
  year          = {{2026}},
  eprint        = {{2606.10154}},
  archivePrefix = {{arXiv}}
}}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="immutable run artifact to publish under runs/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="upload the prepared dataset files",
    )
    args = parser.parse_args()
    run_path = args.run.resolve()
    if not run_path.is_file():
        raise SystemExit(f"run artifact not found: {run_path}")

    corpus_doc = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus = corpus_doc["items"] if isinstance(corpus_doc, dict) else corpus_doc
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="quantsafe-judge-dataset-") as tmp:
        root = Path(tmp)
        (root / "runs").mkdir()
        (root / "README.md").write_text(
            _dataset_card(results, run_path.name),
            encoding="utf-8",
        )
        (root / "judge_corpus.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                for item in corpus
            ),
            encoding="utf-8",
        )
        (root / "judge_verdicts.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (root / "runs" / run_path.name).write_bytes(run_path.read_bytes())

        print(f"Prepared {len(corpus)} rows and run {run_path.name}")
        if args.apply:
            HfApi().upload_folder(
                repo_id=DATASET_ID,
                repo_type="dataset",
                folder_path=str(root),
                commit_message="docs: publish auditable three-guard benchmark",
            )
            print(f"Uploaded to {DATASET_ID}")
        else:
            print("Preview only. Re-run with --apply to publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
