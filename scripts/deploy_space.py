"""scripts/deploy_space.py — push the repo to the HF Space + set secrets.

Uploads the working tree to huggingface.co/spaces/Crusadersk/quantsafe-certifier
(respecting .gitignore-style ignore patterns) and, when the corresponding
environment variables are present, sets the Space secrets the app needs:

    MODAL_ENDPOINT               -> live debate / live screen modal backend URL
    MODAL_TOKEN                  -> bearer token (matches Modal quantsafe-auth)
    GRADIO_CERT_SIGNING_KEY_HEX  -> stable Ed25519 issuer key across restarts
    HF_TOKEN                     -> for gated/Inference-Provider model access

Usage (PowerShell):
    $env:MODAL_ENDPOINT = "https://sahilkadadekar--generate.modal.run"
    $env:MODAL_TOKEN    = "<token>"
    python scripts/deploy_space.py            # upload + set whatever secrets are in env
    python scripts/deploy_space.py --no-upload  # only set secrets
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Crusadersk/quantsafe-certifier"
ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    "__pycache__/*",
    "*.pyc",
    ".git/*",
    ".github/*",
    ".claude/*",
    ".pytest_cache/*",
    ".benchmarks/*",
    # SECURITY: never upload local secret material to the public Space.
    ".modal_token_local.txt",
    ".cert_key_local.txt",
    ".env",
    "*.pem",
    ".playwright-mcp/*",
    "_applog.txt",
    "*.log",
    "scripts/*",  # regen/deploy tooling is dev-only; the Space doesn't run it
]

# Secrets to mirror into the Space when present in the local environment.
SECRET_ENV_VARS = [
    "MODAL_ENDPOINT",
    "MODAL_TOKEN",
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
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=str(ROOT),
            ignore_patterns=IGNORE,
            commit_message="Tier 2-4: SOTA cohort, correctness fixes, gradio 5.50.0",
        )
        print("Upload complete.")

    info = api.space_info(REPO_ID)
    print(f"Space runtime stage: {getattr(getattr(info, 'runtime', None), 'stage', '?')}")
    print(f"Private: {info.private}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
