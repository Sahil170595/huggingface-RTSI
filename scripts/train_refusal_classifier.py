"""Fine-tune a compact semantic refusal classifier and publish it to the Hub.

Training data:
  - WildGuardMix cleaned mirror (balanced refusal/compliance response labels)

External evaluation:
  - XSTest GPT-4 responses with full-compliance/full-refusal labels

The script compares the fine-tuned model with QuantSafe's legacy opener lexicon.
It pushes the best checkpoint only when the external result beats the lexicon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, get_token
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from features import is_refusal  # noqa: E402

BASE_MODEL = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TRAIN_DATASET = "bogdanminko/wildguardmix-cleaned"
EXTERNAL_DATASET = "natolambert/xstest-v2-copy"
DEFAULT_REPO = "Crusadersk/quantsafe-refusal-modernbert"
SEED = 20260613


def _parquet_url(dataset_id: str, split: str) -> str:
    endpoint = (
        "https://datasets-server.huggingface.co/parquet?dataset="
        + quote(dataset_id, safe="")
    )
    response = requests.get(endpoint, timeout=60)
    response.raise_for_status()
    files = response.json()["parquet_files"]
    return next(item["url"] for item in files if item["split"] == split)


def _format_text(prompt: str, response: str) -> str:
    return f"[USER]\n{prompt}\n\n[ASSISTANT]\n{response}"


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "refusal_precision": float(precision),
        "refusal_recall": float(recall),
        "refusal_f1": float(f1),
    }


def _compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return _metrics(labels, predictions)


def _load_frames(smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(_parquet_url(TRAIN_DATASET, "train"))
    train = train.dropna(subset=["prompt", "response", "response_refusal_label"])
    train = train[
        train["response_refusal_label"].isin(["refusal", "compliance"])
    ].copy()
    train["label"] = (train["response_refusal_label"] == "refusal").astype(int)
    train["text"] = [
        _format_text(str(prompt), str(response))
        for prompt, response in zip(train["prompt"], train["response"])
    ]
    train = train[["text", "label"]]

    external = pd.read_parquet(_parquet_url(EXTERNAL_DATASET, "gpt4"))
    external = external[
        external["final_label"].isin(["1_full_compliance", "2_full_refusal"])
    ].copy()
    external["label"] = (external["final_label"] == "2_full_refusal").astype(int)
    external["text"] = [
        _format_text(str(prompt), str(response))
        for prompt, response in zip(external["prompt"], external["completion"])
    ]
    external = external[["prompt", "completion", "text", "label"]]

    if smoke:
        train = (
            train.groupby("label", group_keys=False)
            .sample(n=64, random_state=SEED)
            .reset_index(drop=True)
        )
        external = (
            external.groupby("label", group_keys=False)
            .sample(n=32, random_state=SEED)
            .reset_index(drop=True)
        )
    return train, external


def _model_card(repo_id: str, metrics: dict, n_train: int, n_external: int) -> str:
    external = metrics["external_xstest"]
    lexical = metrics["lexical_xstest"]
    return f"""---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: {BASE_MODEL}
datasets:
- {TRAIN_DATASET}
- {EXTERNAL_DATASET}
tags:
- refusal
- safety
- quantization
- modernbert
- build-small
---

# QuantSafe Refusal ModernBERT

A compact binary classifier used by QuantSafe Certifier to distinguish semantic
refusals from compliant responses. Label `1` is `refusal`; label `0` is
`compliance`.

## Training

- Base: `{BASE_MODEL}` pinned to `{BASE_REVISION}`
- Training: {n_train:,} balanced WildGuardMix prompt/response pairs
- External test: {n_external:,} unambiguous XSTest GPT-4 responses
- Seed: `{SEED}`

## External XSTest result

| Method | Accuracy | Macro F1 | Refusal F1 |
|---|---:|---:|---:|
| This model | {external['accuracy']:.4f} | {external['macro_f1']:.4f} | {external['refusal_f1']:.4f} |
| QuantSafe legacy opener lexicon | {lexical['accuracy']:.4f} | {lexical['macro_f1']:.4f} | {lexical['refusal_f1']:.4f} |

The model is a refusal detector, not a general-purpose harmfulness classifier.
It should be used as a screening signal rather than as a standalone safety
decision.

## Sources

- [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix), ODC-BY
- [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy)
- [QuantSafe Certifier](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default=str(_ROOT / ".benchmarks" / "refusal-modernbert"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    train_frame, external_frame = _load_frames(args.smoke)
    train_part, validation_part = train_test_split(
        train_frame,
        test_size=0.1,
        random_state=SEED,
        stratify=train_frame["label"],
    )
    split = DatasetDict({
        "train": Dataset.from_pandas(train_part, preserve_index=False),
        "test": Dataset.from_pandas(validation_part, preserve_index=False),
    })

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized = split.map(tokenize, batched=True, remove_columns=["text"])
    external_dataset = Dataset.from_pandas(
        external_frame[["text", "label"]], preserve_index=False
    ).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        num_labels=2,
        id2label={0: "compliance", 1: "refusal"},
        label2id={"compliance": 0, "refusal": 1},
    )
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8 if args.smoke else 16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1 if args.smoke else 2,
        num_train_epochs=1 if args.smoke else 2,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="steps",
        eval_steps=10 if args.smoke else 250,
        save_strategy="steps",
        save_steps=10 if args.smoke else 250,
        logging_steps=5 if args.smoke else 50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    validation_metrics = trainer.evaluate()

    external_prediction = trainer.predict(external_dataset)
    external_labels = external_frame["label"].to_numpy(dtype=np.int64)
    external_predicted = np.argmax(external_prediction.predictions, axis=-1)
    external_metrics = _metrics(external_labels, external_predicted)
    lexical_predicted = external_frame["completion"].map(is_refusal).to_numpy(dtype=np.int64)
    lexical_metrics = _metrics(external_labels, lexical_predicted)
    metrics = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "seed": SEED,
        "n_train_total": len(train_frame),
        "n_external_xstest": len(external_frame),
        "validation": {
            key.removeprefix("eval_"): float(value)
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float))
        },
        "external_xstest": external_metrics,
        "lexical_xstest": lexical_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    if external_metrics["macro_f1"] <= lexical_metrics["macro_f1"]:
        raise SystemExit(
            "External XSTest macro F1 did not beat the lexical baseline; refusing to publish."
        )
    if args.no_push:
        trainer.save_model(str(output_dir / "best"))
        tokenizer.save_pretrained(str(output_dir / "best"))
        return 0

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN or a local Hugging Face login is required to push")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    trainer.model.push_to_hub(args.repo_id, token=token)
    tokenizer.push_to_hub(args.repo_id, token=token)
    api.upload_file(
        path_or_fileobj=json.dumps(metrics, indent=2).encode(),
        path_in_repo="metrics.json",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add external XSTest evaluation",
    )
    api.upload_file(
        path_or_fileobj=_model_card(
            args.repo_id, metrics, len(train_frame), len(external_frame)
        ).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Document training and evaluation",
    )
    print(f"Published https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
