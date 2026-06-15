"""Publish idempotent QuantSafe release warnings to screened model cards."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

START = "<!-- quantsafe-release-gate:start -->"
END = "<!-- quantsafe-release-gate:end -->"

WARNINGS = {
    "Crusadersk/phi-2-gptq-4bit": """
## Deployment safety notice

**QuantSafe release action: `ROUTE`.**

I retrospectively screened the immutable weight revision
[`6385e88d733f...`](https://huggingface.co/Crusadersk/phi-2-gptq-4bit/tree/6385e88d733fe95b67dc6d18f264b83c6462e681)
against its matched FP16 baseline. The frozen screen reports **RTSI 0.6199
(`HIGH`)** and a raw refusal-rate shift from **91% to 1% (-90 percentage
points)**.

This result does not prove that the checkpoint is unsafe. It is the publisher's
release-gate decision: **do not deploy this revision without direct safety
evaluation; route to a safer baseline when the deployment cannot absorb that
evaluation.**

[Open the QuantSafe evidence](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier)
· [Read the methodology](https://arxiv.org/abs/2606.10154)

_This README-only notice does not modify the screened model weights._
""".strip(),
    "Crusadersk/qwen2.5-1.5b-gptq-4bit": """
## Deployment safety notice

**QuantSafe release action: `ROUTE`.**

I retrospectively screened the immutable weight revision
[`4e1c7d4d78a3...`](https://huggingface.co/Crusadersk/qwen2.5-1.5b-gptq-4bit/tree/4e1c7d4d78a3fbb82742207baa7ac305bd836cb5)
against its matched FP16 baseline. The frozen screen reports **RTSI 0.7864
(`HIGH`, the matrix maximum)** and a raw refusal-rate shift from **99% to 47%
(-52 percentage points)**.

This result does not prove that the checkpoint is unsafe. It is the publisher's
release-gate decision: **do not deploy this revision without direct safety
evaluation; route to a safer baseline when the deployment cannot absorb that
evaluation.**

[Open the QuantSafe evidence](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier)
· [Read the methodology](https://arxiv.org/abs/2606.10154)

_This README-only notice does not modify the screened model weights._
""".strip(),
}


def _replace_or_insert(readme: str, warning: str) -> str:
    block = f"{START}\n{warning}\n{END}"
    if START in readme and END in readme:
        prefix, remainder = readme.split(START, 1)
        _, suffix = remainder.split(END, 1)
        return f"{prefix}{block}{suffix}"

    lines = readme.splitlines()
    title_index = next(
        (index for index, line in enumerate(lines) if line.startswith("# ")),
        None,
    )
    if title_index is None:
        raise ValueError("model card has no H1 heading")
    lines[title_index + 1:title_index + 1] = ["", block]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="upload the updated cards; otherwise perform a read-only preview",
    )
    args = parser.parse_args()
    api = HfApi()

    with tempfile.TemporaryDirectory(prefix="quantsafe-model-cards-") as tmp:
        for repo_id, warning in WARNINGS.items():
            cached = Path(hf_hub_download(repo_id=repo_id, filename="README.md"))
            updated = _replace_or_insert(cached.read_text(encoding="utf-8"), warning)
            target = Path(tmp) / repo_id.replace("/", "--") / "README.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(updated, encoding="utf-8")
            print(f"{repo_id}: prepared {len(updated)} UTF-8 bytes")
            if args.apply:
                api.upload_file(
                    path_or_fileobj=str(target),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    commit_message="docs: publish QuantSafe ROUTE decision",
                )
                print(f"{repo_id}: uploaded")

    if not args.apply:
        print("Preview only. Re-run with --apply to publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
