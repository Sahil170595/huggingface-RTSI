"""scripts/deploy_space.py — push the repo to the HF Space + set secrets.

Uploads the working tree to the official Build Small organization Space
(respecting .gitignore-style ignore patterns) and, when the corresponding
environment variables are present, sets the Space secrets the app needs:

    MODAL_ENDPOINT               -> live debate / live screen modal backend URL
    MODAL_TOKEN                  -> bearer token (matches Modal quantsafe-auth)
    OPENBMB_API_KEY              -> MiniCPM debate / benchmark provider key
    GRADIO_CERT_SIGNING_KEY_HEX  -> stable Ed25519 issuer key across restarts
    HF_TOKEN                     -> for gated/Inference-Provider model access

Usage (PowerShell):
    # The active Hugging Face token must have write access to build-small-hackathon.
    $env:MODAL_ENDPOINT = "https://sahilkadadekar--generate.modal.run"
    $env:MODAL_TOKEN    = "<token>"
    python scripts/deploy_space.py            # upload + set whatever secrets are in env
    python scripts/deploy_space.py --no-upload  # only set secrets
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "build-small-hackathon/quantsafe-certifier"
ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    "__pycache__/*",
    "*.pyc",
    ".git/*",
    ".github/*",
    ".claude/*",
    ".pytest_cache/*",
    ".mypy_cache/*",
    ".benchmarks/*",
    # SECURITY: never upload local secret material to the public Space.
    ".modal_token_local.txt",
    ".cert_key_local.txt",
    ".env",
    "*.pem",
    ".playwright-mcp/*",
    ".playwright-cli/*",
    "quantsafe-live.png",
    ".history/*",
    ".ruff_cache/*",
    "output/*",
    "scripts/_prospective_cache/*",  # raw model completions to harmful probes — never publish
    # SECURITY: internal competitive-strategy docs never ship in the Space.
    # AGENT_TRACE.md is intentionally public for the Sharing is Caring badge.
    "HACKATHON_BRIEF.md",
    "HACKATHON_ORG_PAGE.md",
    # The org token can commit source but cannot negotiate LFS uploads. Demo
    # media is uploaded separately through the authenticated Hugging Face UI.
    "demo/*.mp4",
    "demo/*.webm",
    "social/*",
    "_applog.txt",
    "*.log",
    "scripts/deploy_space.py",
    "scripts/regen_debate.py",
    "scripts/regen_judges.py",
    "scripts/regen_validation.py",
    "scripts/train_refusal_classifier.py",
    "scripts/publish_judge_benchmark.py",
    "scripts/publish_release_warnings.py",
    # research/eval scripts — not part of the running Space app
    "scripts/eval_external_judges.py",
    "scripts/eval_openbmb_minicpm.py",
    "scripts/prospective_modal.py",
    "scripts/prospective_score.py",
]

# Secrets to mirror into the Space when present in the local environment.
SECRET_ENV_VARS = [
    "MODAL_ENDPOINT",
    "MODAL_TOKEN",
    "OPENBMB_API_KEY",
    "GRADIO_CERT_SIGNING_KEY_HEX",
    "HF_TOKEN",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true", help="only set secrets")
    args = ap.parse_args()

    api = HfApi()

    for name in SECRET_ENV_VARS:
        val = os.environ.get(name)
        if not val:
            print(f"  (skip secret {name}: not in env)")
            continue
        api.add_space_secret(repo_id=REPO_ID, key=name, value=val)
        print(f"  set Space secret {name} ({len(val)} chars)")

    if not args.no_upload:
        print(f"Uploading working tree to {REPO_ID} ...")
        commit = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=str(ROOT),
            ignore_patterns=IGNORE,
            commit_message="Deploy audited QuantSafe Certifier",
            create_pr=True,
        )
        print(f"Upload complete: {commit.pr_url or commit.commit_url}")

    info = api.space_info(REPO_ID)
    print(f"Space runtime stage: {getattr(getattr(info, 'runtime', None), 'stage', '?')}")
    print(f"Private: {info.private}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
