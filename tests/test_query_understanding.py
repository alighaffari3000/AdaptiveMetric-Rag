"""Phase 4: multi-label routing, conversation memory, and multi-query fusion."""

import asyncio
import json
import uuid

import pytest

from app import database
from app.index import index as chunk_index
from app.models import AppSettings, QueryAnalysis
from app.retrieval import RetrievalResult, analyze_query, blend_weights, embed, fuse_results, retrieve, route
from app.rewrite import (
    clear_cache,
    context_terms,
    needs_context,
    parse_rewrite,
    plan_query,
    rule_plan,
)
from app.text import tokenize


# --- the router now scores every intent instead of taking the first branch ---


def test_a_compound_question_carries_every_intent_it_states():
    """«چرا مبلغ ... در سال ...» is causal, numeric and temporal at once.

    The if/elif chain filed it as temporal alone and threw the rest away.
    """
    analysis = analyze_query("چرا مبلغ قرارداد ۱۳۷ در سال ۱۴۰۳ تغییر کرد؟")
    assert set(analysis.intents) >= {"causal", "numeric_fact", "temporal_fact"}
    assert sum(analysis.intent_shares.values()) == pytest.approx(1.0, abs=1e-3)


def test_a_compound_question_mixes_the_weight_vectors():
    single = analyze_query("مبلغ قرارداد چقدر است؟")
    compound = analyze_query("چرا مبلغ قرارداد ۱۳۷ در سال ۱۴۰۳ تغییر کرد؟")
    # The numeric reading is diluted by the causal and temporal ones, and the
    # signals those need are raised in exchange.
    assert compound.weights["numeric"] < single.weights["numeric"]
    assert compound.weights["temporal"] > single.weights["temporal"]
    assert compound.weights["dense"] > single.weights["dense"]
    assert sum(compound.weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_a_single_intent_question_is_routed_as_before():
    analysis = analyze_query("قرارداد شماره ۱۳۷ چه زمانی تمام می‌شود؟")
    assert analysis.intent == "temporal_fact"
    assert analysis.intents == ["temporal_fact"]
    assert analysis.weights == pytest.approx(
        {"dense": .16, "bm25": .10, "keyword": .10, "entity": .22,
         "numeric": .07, "temporal": .28, "metadata": .07}, abs=1e-6
    )


def test_browsing_still_overrides_the_rest():
    analysis = analyze_query("از متن کتاب برام بنویس")
    assert analysis.intent == "document_browse"
    assert analysis.intent_shares["document_browse"] > .5


def test_a_question_with_no_telling_words_falls_back_to_one_intent():
    shares = route("the northwind platform", [], [], [])
    assert shares == {"exact_fact": 1.0}


def test_blending_one_intent_reproduces_its_own_vector():
    assert blend_weights({"causal": 1.0}) == pytest.approx(analyze_query("چرا فسخ شد؟").weights, abs=1e-6)


# --- conversation memory ---


def test_a_follow_up_is_recognised_and_a_full_question_is_not():
    assert needs_context("و مبلغش چقدر بود؟")
    assert needs_context("چرا؟")
    assert needs_context("And what about the onboarding fee?")
    assert not needs_context("قرارداد شماره ۱۳۷ چه زمانی تمام می‌شود؟")


def test_the_topic_of_the_conversation_is_carried_into_the_question():
    history = [
        {"role": "user", "content": "قرارداد شماره ۱۳۷ بین چه شرکت‌هایی امضا شد؟"},
        {"role": "assistant", "content": "قرارداد شماره ۱۳۷ میان شرکت آفتاب و شرکت سپهر امضا شد."},
    ]
    plan = rule_plan("و مبلغش چقدر بود؟", history, turns=4)
    assert plan.source == "rules"
    assert plan.rewritten
    assert "قرارداد" in plan.question and "137" in tokenize(plan.question)
    # The words the person chose are still there, and still first.
    assert plan.question.startswith("و مبلغش چقدر بود؟")


def test_a_self_contained_question_is_left_alone():
    history = [{"role": "user", "content": "قرارداد شماره ۱۳۷ بین چه شرکت‌هایی امضا شد؟"}]
    plan = rule_plan("مهلت اعلام تمدید قرارداد چند روز پیش از پایان است؟", history, turns=4)
    assert not plan.rewritten
    assert plan.source == "verbatim"


def test_questions_are_preferred_over_answers_as_context():
    history = [
        {"role": "user", "content": "قرارداد شماره ۱۳۷ چیست؟"},
        {"role": "assistant", "content": "توضیح مفصل درباره نگهداری زیرساخت و پشتیبانی فنی."},
    ]
    terms = context_terms(history, "و مبلغش؟", turns=4)
    assert terms.index("137") < terms.index("نگهداری")


def test_only_the_configured_number_of_turns_is_used():
    history = [{"role": "user", "content": "اولین موضوع دربارهٔ سپهر"},
               {"role": "assistant", "content": "پاسخ اول"},
               {"role": "user", "content": "دومین موضوع دربارهٔ آفتاب"},
               {"role": "assistant", "content": "پاسخ دوم"}]
    terms = context_terms(history, "چرا؟", turns=2)
    assert "افتاب" in terms and "سپهر" not in terms


# --- the LLM rewriter's output is never trusted as given ---


def test_a_fenced_json_rewrite_is_read():
    raw = '```json\n{"standalone": "مبلغ قرارداد ۱۳۷ چقدر است؟", "queries": ["مبلغ کل قرارداد ۱۳۷"]}\n```'
    plan = parse_rewrite(raw, "و مبلغش چقدر بود؟")
    assert plan is not None
    assert plan.question == "مبلغ قرارداد ۱۳۷ چقدر است؟"
    assert plan.variants == ["مبلغ کل قرارداد ۱۳۷"]


def test_prose_around_the_json_is_tolerated():
    plan = parse_rewrite('Sure! {"standalone": "the onboarding fee in MSA-2214"} Hope that helps.', "x")
    assert plan is not None and plan.question == "the onboarding fee in MSA-2214"


@pytest.mark.parametrize("raw", ["", "no json here", "{}", '{"standalone": ""}', '{"standalone": 12}', "[1, 2]"])
def test_unusable_rewrites_are_refused(raw):
    assert parse_rewrite(raw, "و مبلغش چقدر بود؟") is None


def test_a_local_provider_never_calls_a_model_and_still_rewrites():
    clear_cache()
    settings = AppSettings(provider="local")
    history = [{"role": "user", "content": "درباره قرارداد شماره ۱۳۷ بگو"}]
    plan = asyncio.run(plan_query(settings, "کی تموم می‌شه؟", history, conversation_id="c1"))
    assert plan.source == "rules"
    assert "137" in tokenize(plan.question)


def test_rewriting_can_be_switched_off():
    clear_cache()
    settings = AppSettings(provider="local", enable_query_rewrite=False)
    history = [{"role": "user", "content": "درباره قرارداد شماره ۱۳۷ بگو"}]
    plan = asyncio.run(plan_query(settings, "کی تموم می‌شه؟", history, conversation_id="c1"))
    assert not plan.rewritten


def test_the_cache_distinguishes_the_same_question_asked_twice():
    """A cached rewrite keyed on the message alone would answer about the wrong contract."""
    clear_cache()
    settings = AppSettings(provider="local")
    first = asyncio.run(plan_query(settings, "و مبلغش؟",
                                   [{"role": "user", "content": "قرارداد ۱۳۷ چیست؟"}], "c1"))
    second = asyncio.run(plan_query(settings, "و مبلغش؟",
                                    [{"role": "user", "content": "قرارداد ۲۰۸ چیست؟"}], "c1"))
    assert "137" in tokenize(first.question)
    assert "208" in tokenize(second.question)
    repeat = asyncio.run(plan_query(settings, "و مبلغش؟",
                                    [{"role": "user", "content": "قرارداد ۱۳۷ چیست؟"}], "c1"))
    assert repeat.question == first.question
    assert repeat.source.endswith("-cached")


# --- multi-query fusion ---


def _result(ids: list[str]) -> RetrievalResult:
    analysis = QueryAnalysis(intent="exact_fact", language="fa", weights={"dense": 1.0},
                             entities=[], numbers=[], temporal_terms=[])
    chunks = [{"id": cid, "content": cid, "score": 1.0 - i * .1, "standout": 1.0,
               "features": {"dense": .5}} for i, cid in enumerate(ids)]
    return RetrievalResult(chunks, analysis, .5, False, 2.0, "rank")


def test_a_chunk_ranked_by_every_phrasing_wins():
    """Agreement across phrasings beats being first in only one of them."""
    fused = fuse_results([_result(["a", "b", "c"]), _result(["d", "b", "e"])], context_count=3)
    assert [chunk["id"] for chunk in fused.chunks][0] == "b"


def test_fusing_one_ranking_changes_nothing():
    single = _result(["a", "b"])
    assert fuse_results([single], context_count=5) is single


def test_a_phrasing_that_found_nothing_does_not_dilute_the_rest():
    fused = fuse_results([_result([]), _result(["a", "b"])], context_count=5)
    assert [chunk["id"] for chunk in fused.chunks] == ["a", "b"]


# --- end to end: the same question in two scripts must retrieve the same thing ---


@pytest.fixture
def small_corpus(fresh_db):
    rows = [
        ("قرارداد شماره ۱۳۷ در تاریخ ۱۴۰۵/۰۱/۱۵ امضا شد و مبلغ کل آن ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است.", "contract-137.fa.txt"),
        ("قرارداد شماره ۲۰۸ مربوط به خدمات ابری است و مبلغ آن ۹۰۰٬۰۰۰٬۰۰۰ ریال است.", "contract-208.fa.txt"),
        ("The Northwind agreement charges a one-time onboarding fee of EUR 24,000.", "msa-northwind.en.txt"),
    ]
    now = "2026-01-01T00:00:00+00:00"
    with database.connect() as db:
        for content, name in rows:
            document_id = uuid.uuid4().hex
            db.execute("INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
                       (document_id, name, "text/plain", len(content), 1, now, "{}"))
            db.execute(
                "INSERT INTO chunks(id,document_id,position,page,section,content,vector,tokens,metadata) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, document_id, 0, None, None, content,
                 database.encode_vector(embed(content)),
                 json.dumps(tokenize(content), ensure_ascii=False), "{}"),
            )
    chunk_index.invalidate()
    yield


def test_persian_and_latin_digits_retrieve_the_same_thing(small_corpus):
    persian = retrieve("مبلغ قرارداد شماره ۱۳۷ چقدر است؟", context_count=3)
    latin = retrieve("مبلغ قرارداد شماره 137 چقدر است؟", context_count=3)
    assert [chunk["id"] for chunk in persian.chunks] == [chunk["id"] for chunk in latin.chunks]
    assert persian.chunks[0]["document_name"] == "contract-137.fa.txt"
    assert persian.chunks[0]["features"]["numeric"] == latin.chunks[0]["features"]["numeric"] == 1.0


def test_a_written_date_matches_the_numeric_one(small_corpus):
    result = retrieve("چه چیزی در ۱۵ فروردین ۱۴۰۵ امضا شد؟", context_count=3)
    assert result.chunks[0]["document_name"] == "contract-137.fa.txt"
    assert result.chunks[0]["features"]["temporal"] == 1.0


def test_an_english_follow_up_after_persian_turns_is_still_english(small_corpus):
    """Persian topic words are for retrieval; they must not switch the reply's language."""
    from app.search import search

    clear_cache()
    history = [{"role": "user", "content": "قرارداد شماره ۲۰۸ درباره چیست؟"},
               {"role": "assistant", "content": "قرارداد ۲۰۸ مربوط به خدمات ابری است."}]
    result = asyncio.run(search(AppSettings(provider="local"), "And the amount?",
                                history=history, conversation_id="c9"))
    assert result.analysis.rewritten_from == "And the amount?"
    assert result.analysis.language == "en"


def test_a_follow_up_reaches_the_document_its_own_words_cannot(small_corpus):
    history = [{"role": "user", "content": "قرارداد شماره ۲۰۸ درباره چیست؟"},
               {"role": "assistant", "content": "قرارداد ۲۰۸ مربوط به خدمات ابری است."}]
    bare = retrieve("و مبلغش چقدر بود؟", context_count=3)
    plan = rule_plan("و مبلغش چقدر بود؟", history, turns=4)
    rewritten = retrieve(plan.question, context_count=3)
    assert rewritten.chunks[0]["document_name"] == "contract-208.fa.txt"
    assert bare.chunks[0]["document_name"] != "contract-208.fa.txt"
