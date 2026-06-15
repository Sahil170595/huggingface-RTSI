"""Immutable Hugging Face model revisions used by local and Modal loaders.

Pinning every downloaded model to a commit SHA makes the screening and debate
runtime reproducible even when an upstream model repository changes its main
branch.
"""

from __future__ import annotations


MODEL_REVISIONS: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    "tiiuae/Falcon3-3B-Instruct": "411bb94318f94f7a5735b77109f456b1e74b42a1",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "HuggingFaceTB/SmolLM3-3B": "a07cc9a04f16550a088caea529712d1d335b0ac1",
    "Qwen/Qwen3Guard-Gen-0.6B": "fada3b2f655b89601929198343c94cd2f64d93cc",
    "ibm-granite/granite-guardian-3.3-8b": "b3421eda4ba6fc9f9a71121d7e62de08827469a4",
    "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3": "8fdc246ba3d56db9c469d534233b9f582d3afafa",
    "openbmb/MiniCPM4.1-8B": "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3",
    "Crusadersk/quantsafe-refusal-modernbert": "b34061f964619a5b6e0ff24be45a428124fa36bc",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
}


def model_revision(model_id: str) -> str:
    """Return the audited commit SHA for a supported model."""
    try:
        return MODEL_REVISIONS[model_id]
    except KeyError as exc:
        raise ValueError(f"No pinned revision configured for model {model_id!r}") from exc
