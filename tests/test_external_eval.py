from __future__ import annotations

from scripts import eval_external_judges


def test_external_dataset_revision_is_immutable_sha():
    revision = eval_external_judges.DATASET_REVISION
    assert len(revision) == 40
    assert set(revision) <= set("0123456789abcdef")


def test_corpus_digest_is_deterministic_and_order_sensitive():
    corpus = [
        {"prompt": "p1", "response": "r1", "expected": "safe"},
        {"prompt": "p2", "response": "r2", "expected": "unsafe"},
    ]
    digest = eval_external_judges.corpus_sha256(corpus)
    assert digest == eval_external_judges.corpus_sha256(list(corpus))
    assert digest != eval_external_judges.corpus_sha256(list(reversed(corpus)))
    assert len(digest) == 64


def test_run_eval_binds_revision_and_corpus_digest():
    corpus = [
        {"prompt": "p1", "response": "r1", "expected": "safe"},
        {"prompt": "p2", "response": "r2", "expected": "unsafe"},
    ]
    result = eval_external_judges.run_eval(
        corpus,
        post_judge_fn=eval_external_judges._make_mock_post_judge(1),
        dry_run=True,
    )
    assert result["dataset_revision"] == eval_external_judges.DATASET_REVISION
    assert result["corpus_sha256"] == eval_external_judges.corpus_sha256(corpus)
