"""Tests for rank fusion, the semantic weight profile, and second-stage reranking."""

import asyncio

import numpy as np
import pytest

from app import rerank
from app.models import AppSettings
from app.rerank import RerankUnavailable, blend, build_llm_prompt, parse_llm_scores, rerank_scores
from app.retrieval import _weighted_rrf, _weighted_sum, apply_semantic_profile, score_confidence

EVEN = {"dense": .5, "bm25": .5}


def test_rank_fusion_is_unchanged_by_rescaling_a_signal():
    """The defect rank fusion exists to fix.

    Both dense arrays below rank the candidates identically; only the spread
    differs, exactly as a semantic cosine differs from a min-max normalised
    BM25. The weighted sum reaches opposite conclusions on the two, while rank
    fusion reads only the ordering and answers the same both times.
    """
    narrow = np.array([0.71, 0.70, 0.69])   # a semantic cosine: a tight, high band
    wide = np.array([1.00, 0.50, 0.10])     # the same order, spread far wider
    bm25 = np.array([0.00, 0.45, 0.20])
    weights = {"dense": .5, "bm25": .5}

    linear_narrow = int(np.argmax(_weighted_sum({"dense": narrow, "bm25": bm25}, weights)))
    linear_wide = int(np.argmax(_weighted_sum({"dense": wide, "bm25": bm25}, weights)))
    assert linear_narrow != linear_wide, "the weighted sum follows whichever signal spreads wider"

    fused_narrow = _weighted_rrf({"dense": narrow, "bm25": bm25}, weights)
    fused_wide = _weighted_rrf({"dense": wide, "bm25": bm25}, weights)
    assert np.allclose(fused_narrow, fused_wide), "rank fusion depends only on the ordering"


def test_a_zero_means_the_signal_says_nothing_about_that_chunk():
    """Worth stating because it is the one place ordering is not the whole story.

    A zero is read as "this signal has no opinion here", not as "ranked last".
    That is what makes sparse signals usable: a chunk that never mentions the
    entity should not be ranked on the entity signal at all. It also means a
    dense score clamped to zero drops out of the dense ranking rather than
    trailing it, which costs that chunk nothing it would otherwise have had.
    """
    present = _weighted_rrf({"entity": np.array([0.8, 0.4, 0.0])}, {"entity": 1.0})
    assert present[2] == 0.0
    assert present[0] > present[1] > present[2]


def test_rank_fusion_respects_the_intent_weights():
    dense = np.array([1.0, 0.9, 0.8])
    bm25 = np.array([0.1, 0.2, 0.3])
    dense_led = _weighted_rrf({"dense": dense, "bm25": bm25}, {"dense": .9, "bm25": .1})
    lexical_led = _weighted_rrf({"dense": dense, "bm25": bm25}, {"dense": .1, "bm25": .9})
    assert int(np.argmax(dense_led)) == 0
    assert int(np.argmax(lexical_led)) == 2


def test_a_signal_that_separates_nothing_does_not_vote():
    varying = np.array([0.9, 0.5, 0.1])
    flat = np.array([0.4, 0.4, 0.4])
    with_flat = _weighted_rrf({"dense": varying, "bm25": flat}, EVEN)
    without_flat = _weighted_rrf({"dense": varying}, {"dense": .5})
    assert np.allclose(with_flat, without_flat), "a constant signal carries no ordering"


def test_an_all_zero_signal_does_not_vote():
    varying = np.array([0.9, 0.5, 0.1])
    zeros = np.zeros(3)
    assert np.allclose(
        _weighted_rrf({"dense": varying, "entity": zeros}, {"dense": .5, "entity": .5}),
        _weighted_rrf({"dense": varying}, {"dense": .5}),
    )


def test_a_partially_zero_signal_still_votes():
    """Zero versus non-zero is real information: only some chunks mention the entity."""
    dense = np.array([0.1, 0.5, 0.9])
    entity = np.array([1.0, 0.0, 0.0])
    fused = _weighted_rrf({"dense": dense, "entity": entity}, {"dense": .2, "entity": .8})
    assert int(np.argmax(fused)) == 0


def test_a_single_candidate_scores_at_the_top_rather_than_zero():
    """Regression: every signal is constant across one candidate, so all of them
    are skipped. Returning zero made the evidence gate refuse a good answer."""
    fused = _weighted_rrf({"dense": np.array([0.42]), "bm25": np.array([1.0])}, EVEN)
    assert fused.tolist() == [1.0]


def test_fusion_scores_stay_within_the_unit_range():
    dense = np.array([0.9, 0.5, 0.1])
    bm25 = np.array([0.2, 0.9, 0.4])
    fused = _weighted_rrf({"dense": dense, "bm25": bm25}, EVEN)
    assert fused.max() <= 1.0 + 1e-9 and fused.min() >= 0.0


def test_semantic_profile_raises_dense_and_keeps_the_rest_proportional():
    weights = {"dense": .16, "bm25": .10, "entity": .22, "temporal": .28,
               "keyword": .10, "numeric": .07, "metadata": .07}
    adjusted = apply_semantic_profile(weights, .85)
    assert adjusted["dense"] == .85
    assert sum(adjusted.values()) == pytest.approx(1.0)
    # temporal was the largest of the others and stays the largest of the others.
    others = {k: v for k, v in adjusted.items() if k != "dense"}
    assert max(others, key=others.get) == "temporal"
    assert adjusted["temporal"] / adjusted["bm25"] == pytest.approx(.28 / .10)


def test_semantic_profile_leaves_an_already_dense_led_intent_alone():
    weights = {"dense": .92, "bm25": .08}
    assert apply_semantic_profile(weights, .85) == weights


def test_confidence_uses_the_reranker_verdict_when_there_is_one():
    chunks = [{"content": "قرارداد شماره ۱۳۷ مبلغ کل دارد", "score": 1.0, "rerank_score": 0.95}]
    high = score_confidence(chunks, ["قرارداد", "مبلغ"], standout=0.5)
    chunks[0]["rerank_score"] = 0.05
    low = score_confidence(chunks, ["قرارداد", "مبلغ"], standout=0.5)
    assert high > low
    assert high > 0.9 and low < 0.5


def test_confidence_falls_back_to_how_far_the_best_chunk_stands_out():
    chunks = [{"content": "قرارداد شماره ۱۳۷", "score": 1.0}]
    strong = score_confidence(chunks, ["قرارداد"], standout=4.0)
    weak = score_confidence(chunks, ["قرارداد"], standout=0.5)
    assert strong > weak
    assert score_confidence([], ["قرارداد"], standout=4.0) == 0.0


def test_confidence_drops_when_the_question_is_barely_covered():
    covered = [{"content": "قرارداد شماره ۱۳۷ مبلغ کل دو میلیارد", "score": 1.0}]
    uncovered = [{"content": "متن نامرتبط درباره چیز دیگری", "score": 1.0}]
    terms = ["قرارداد", "مبلغ", "۱۳۷"]
    assert score_confidence(covered, terms, standout=3.0) > score_confidence(uncovered, terms, standout=3.0)


def test_llm_rerank_prompt_numbers_every_passage():
    system, user = build_llm_prompt("مبلغ قرارداد چقدر است؟", ["passage one", "passage two"])
    assert "[1] passage one" in user and "[2] passage two" in user
    assert "all 2 passages" in user
    assert "different languages" in system, "cross-language pairs must not be penalised"


def test_rerank_scores_parse_from_fenced_json():
    raw = '```json\n{"scores": [{"id": 1, "score": 9}, {"id": 2, "score": 3}]}\n```'
    assert parse_llm_scores(raw, 2) == [0.9, 0.3]


def test_rerank_scores_survive_surrounding_prose_and_clamp_out_of_range_values():
    raw = 'Here are the scores:\n{"scores": [{"id": 1, "score": 42}, {"id": 2, "score": -5}]}\nDone.'
    assert parse_llm_scores(raw, 2) == [1.0, 0.0]


def test_unscored_passages_default_to_zero_rather_than_failing():
    assert parse_llm_scores('{"scores": [{"id": 2, "score": 8}]}', 3) == [0.0, 0.8, 0.0]


@pytest.mark.parametrize("raw", ["not json at all", '{"scores": []}', '{"scores": [{"id": 99, "score": 5}]}'])
def test_unusable_rerank_output_is_reported_not_guessed(raw):
    with pytest.raises(RerankUnavailable):
        parse_llm_scores(raw, 2)


def test_blend_keeps_a_share_of_the_fusion_score():
    chunks = [{"id": "a", "score": 1.0, "content": "a"}, {"id": "b", "score": 0.2, "content": "b"}]
    ordered = blend(chunks, [0.0, 1.0], weight=0.7)
    assert [chunk["id"] for chunk in ordered] == ["b", "a"]
    assert ordered[0]["score"] == pytest.approx(.7 * 1.0 + .3 * 0.2)
    assert ordered[0]["fusion_score"] == 0.2 and ordered[0]["rerank_score"] == 1.0


def test_blend_at_zero_weight_keeps_the_fusion_order():
    chunks = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.2}]
    assert [c["id"] for c in blend(chunks, [0.0, 1.0], weight=0.0)] == ["a", "b"]


def test_llm_reranker_refuses_to_run_against_the_local_provider():
    settings = AppSettings(provider="local", rerank_enabled=True, rerank_backend="llm")
    with pytest.raises(RerankUnavailable, match="local extractive"):
        asyncio.run(rerank_scores(settings, "q", [{"content": "c"}]))


def test_unknown_rerank_backend_is_reported():
    settings = AppSettings(provider="gemini").model_copy(update={"rerank_backend": "nope"})
    with pytest.raises(RerankUnavailable, match="Unknown rerank backend"):
        asyncio.run(rerank_scores(settings, "q", [{"content": "c"}]))


def test_empty_shortlist_needs_no_reranker():
    assert asyncio.run(rerank_scores(AppSettings(provider="gemini"), "q", [])) == []


def test_cross_encoder_scores_are_scaled_into_the_unit_range(monkeypatch):
    monkeypatch.setattr(rerank, "_score_cross_encoder", lambda model, query, passages: [-4.0, 2.0, 8.0])
    settings = AppSettings(rerank_backend="cross-encoder", rerank_model="fake")
    scores = asyncio.run(rerank_scores(settings, "q", [{"content": c} for c in "abc"]))
    assert scores == [0.0, 0.5, 1.0]


def test_cross_encoder_ties_become_neutral_scores(monkeypatch):
    monkeypatch.setattr(rerank, "_score_cross_encoder", lambda model, query, passages: [1.0, 1.0])
    settings = AppSettings(rerank_backend="cross-encoder", rerank_model="fake")
    assert asyncio.run(rerank_scores(settings, "q", [{"content": "a"}, {"content": "b"}])) == [0.5, 0.5]
