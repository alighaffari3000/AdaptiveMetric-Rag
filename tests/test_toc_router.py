"""The zero-shot table-of-contents router.

Every test here runs against a stub completion, so nothing reaches a network.
The properties under test are the two the STAIR paper's numbers depend on: a
heading the model invents is never acted on, and a router that fails costs
nothing because the search widens back to everything.
"""

import asyncio
import json

import pytest

from app import search as search_module
from app import toc_router
from app.models import AppSettings
from app.toc_router import (
    RouterUnavailable,
    constrain,
    eligible,
    matching_positions,
    normalize_heading,
    parse_sections,
    route,
    toc_from_rows,
)

TOC = [
    {"path": "فصل دوم — مرخصی > مرخصی استحقاقی", "title": "مرخصی استحقاقی", "depth": 2},
    {"path": "فصل دوم — مرخصی > مرخصی استعلاجی", "title": "مرخصی استعلاجی", "depth": 2},
    {"path": "فصل سوم — جبران خدمات > پاداش عملکرد", "title": "پاداش عملکرد", "depth": 2},
]


def settings(**overrides) -> AppSettings:
    base = {"provider": "openai", "api_key": "k", "toc_router_enabled": True,
            "toc_router_min_chunks": 2}
    base.update(overrides)
    return AppSettings(**base)


def stub_completion(monkeypatch, reply, calls=None):
    """Replace the provider call; record that it happened and what it returned."""
    from app import providers

    async def fake(settings_, system, user, model="", temperature=0.0, max_tokens=1024):
        if calls is not None:
            calls.append({"system": system, "user": user})
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(providers, "complete", fake)


def run(coro):
    return asyncio.run(coro)


def test_a_valid_reply_selects_those_sections(monkeypatch):
    stub_completion(monkeypatch, json.dumps({"sections": ["فصل دوم — مرخصی > مرخصی استعلاجی"]}))
    chosen = run(route(settings(), "سقف مرخصی استعلاجی چند روز است؟", TOC))
    assert chosen == ["فصل دوم — مرخصی > مرخصی استعلاجی"]


def test_an_invented_heading_is_dropped(monkeypatch):
    """The paper's low hallucination rate comes from constrained output."""
    stub_completion(monkeypatch, json.dumps({"sections": ["فصل نهم — مرخصی زایمان"]}))
    assert run(route(settings(), "مرخصی زایمان چند روز است؟", TOC)) == []


def test_a_half_valid_reply_keeps_only_the_valid_half(monkeypatch):
    stub_completion(monkeypatch, json.dumps({
        "sections": ["فصل سوم — جبران خدمات > پاداش عملکرد", "فصل بیستم — چیزی که وجود ندارد"]
    }))
    assert run(route(settings(), "پاداش چقدر است؟", TOC)) == ["فصل سوم — جبران خدمات > پاداش عملکرد"]


def test_a_retyped_heading_still_matches_across_spelling_variants():
    # Arabic yeh and kaf, an Arabic-Indic digit, and a different dash.
    proposed = ["فصل دوم - مرخصي استعلاجي"]
    assert constrain(proposed, TOC, 3) == ["فصل دوم — مرخصی > مرخصی استعلاجی"]


def test_two_sibling_headings_never_collapse_into_each_other():
    """"استحقاقی" and "استعلاجی" differ by two letters and mean different things."""
    assert constrain(["مرخصی استحقاقی"], TOC, 3) == ["فصل دوم — مرخصی > مرخصی استحقاقی"]
    assert constrain(["مرخصی استعلاجی"], TOC, 3) == ["فصل دوم — مرخصی > مرخصی استعلاجی"]


def test_a_bare_title_resolves_to_its_full_path():
    assert constrain(["پاداش عملکرد"], TOC, 3) == ["فصل سوم — جبران خدمات > پاداش عملکرد"]


def test_the_section_limit_is_respected():
    every = [entry["path"] for entry in TOC]
    assert len(constrain(every, TOC, 2)) == 2


def test_duplicates_in_the_reply_are_collapsed():
    path = TOC[0]["path"]
    assert constrain([path, path, path], TOC, 3) == [path]


def test_an_empty_selection_is_a_valid_answer(monkeypatch):
    stub_completion(monkeypatch, json.dumps({"sections": []}))
    assert run(route(settings(), "چیزی نامرتبط", TOC)) == []


@pytest.mark.parametrize("reply", ["not json at all", "{broken", json.dumps({"other": []})])
def test_an_unusable_reply_raises_rather_than_guessing(monkeypatch, reply):
    stub_completion(monkeypatch, reply)
    with pytest.raises(RouterUnavailable):
        run(route(settings(), "q", TOC))


def test_a_provider_failure_is_reported_as_unavailable(monkeypatch):
    stub_completion(monkeypatch, RuntimeError("connection reset"))
    with pytest.raises(RouterUnavailable):
        run(route(settings(), "q", TOC))


def test_json_wrapped_in_a_code_fence_is_read(monkeypatch):
    stub_completion(monkeypatch, '```json\n{"sections": ["پاداش عملکرد"]}\n```')
    assert run(route(settings(), "q", TOC)) == ["فصل سوم — جبران خدمات > پاداش عملکرد"]


def test_the_prompt_carries_every_heading_and_asks_for_exact_copies(monkeypatch):
    calls = []
    stub_completion(monkeypatch, json.dumps({"sections": []}), calls)
    run(route(settings(), "سؤال", TOC))
    assert len(calls) == 1
    for entry in TOC:
        assert entry["path"] in calls[0]["user"]
    assert "Never invent a section" in calls[0]["system"]


def test_parse_sections_ignores_non_string_entries():
    assert parse_sections(json.dumps({"sections": ["ok", 3, None, "  ", "also ok"]})) == ["ok", "also ok"]


def test_normalize_heading_folds_digits_and_letters():
    assert normalize_heading("ماده ۳") == normalize_heading("ماده 3")
    assert normalize_heading("مرخصي") == normalize_heading("مرخصی")


@pytest.mark.parametrize("overrides,expected", [
    ({}, True),
    ({"toc_router_enabled": False}, False),
    ({"provider": "local", "api_key": ""}, False),
    ({"toc_router_min_chunks": 99}, False),
])
def test_eligibility_gates_the_extra_model_call(overrides, expected):
    assert eligible(settings(**overrides), TOC, chunk_count=10) is expected


def test_a_document_without_a_table_of_contents_is_never_routed():
    assert eligible(settings(), [], chunk_count=1000) is False
    assert eligible(settings(), TOC[:1], chunk_count=1000) is False


def test_matching_positions_finds_every_chunk_that_covers_a_section():
    rows = [
        {"section_path": "فصل دوم — مرخصی > مرخصی استحقاقی | مرخصی استعلاجی"},
        {"section_path": "فصل سوم — جبران خدمات > پاداش عملکرد"},
        {"section_path": ""},
    ]
    assert matching_positions(rows, ["فصل دوم — مرخصی > مرخصی استعلاجی"]) == [0]
    assert matching_positions(rows, ["فصل سوم — جبران خدمات > پاداش عملکرد"]) == [1]
    assert matching_positions(rows, []) == []


def test_choosing_a_parent_section_selects_everything_beneath_it():
    rows = [{"section_path": "فصل دوم — مرخصی > مرخصی استعلاجی"},
            {"section_path": "فصل سوم — جبران خدمات > پاداش عملکرد"}]
    assert matching_positions(rows, ["فصل دوم — مرخصی"]) == [0]


def test_the_contents_are_prefixed_by_document_only_when_several_are_in_scope():
    rows = [{"document_name": "a.md", "section_path": "Guide > Leave"},
            {"document_name": "b.md", "section_path": "Guide > Leave"}]
    both = {entry["path"] for entry in toc_from_rows(rows)}
    assert both == {"a.md > Guide > Leave", "b.md > Guide > Leave"}
    single = {entry["path"] for entry in toc_from_rows(rows, [0])}
    assert single == {"Guide > Leave"}


# --- the pipeline around the router ---------------------------------------

def library(db):
    import json as json_module

    from app.index import index as chunk_index
    from app.retrieval import embed, tokenize

    rows = [
        ("بیست و دو روز کاری در سال", "فصل دوم — مرخصی > مرخصی استحقاقی"),
        ("حداکثر هشت روز در سال با گواهی پزشک", "فصل دوم — مرخصی > مرخصی استعلاجی"),
        ("تا سقف دو ماه حقوق پایه", "فصل سوم — جبران خدمات > پاداش عملکرد"),
    ]
    db.execute("INSERT INTO documents VALUES('d','guide.md','text/markdown',10,3,'2026-01-01','{}')")
    for position, (content, path) in enumerate(rows):
        db.execute(
            "INSERT INTO chunks(id,document_id,position,page,section,section_path,content,vector,tokens,metadata) "
            "VALUES(?,'d',?,NULL,?,?,?,?,?,'{}')",
            (f"c{position}", position, path.split(" > ")[-1], path, content,
             db.encode_vector(embed(content)), json_module.dumps(tokenize(content))),
        )
    chunk_index.invalidate()


def test_routing_narrows_the_search_to_the_chosen_section(fresh_db, monkeypatch):
    library(fresh_db)
    stub_completion(monkeypatch, json.dumps({"sections": ["فصل دوم — مرخصی > مرخصی استعلاجی"]}))
    result = run(search_module.search(settings(), "سقف مرخصی استعلاجی چند روز است؟"))
    assert result.routed_sections == ["فصل دوم — مرخصی > مرخصی استعلاجی"]
    assert [chunk["section_path"] for chunk in result.chunks] == ["فصل دوم — مرخصی > مرخصی استعلاجی"]
    assert "route" in result.timings_ms


def unrouted(question):
    """What the pipeline returns with the router switched off entirely."""
    return run(search_module.search(settings(toc_router_enabled=False), question))


def test_a_hallucinated_section_falls_back_to_the_whole_library(fresh_db, monkeypatch):
    """A router that invents a heading must cost nothing, not lose the answer."""
    library(fresh_db)
    question = "پاداش عملکرد چقدر است؟"
    plain = unrouted(question)
    expected_ids = [chunk["id"] for chunk in plain.chunks]
    stub_completion(monkeypatch, json.dumps({"sections": ["فصل نهم — چیزی که وجود ندارد"]}))
    result = run(search_module.search(settings(), question))
    assert result.routed_sections == []
    assert [chunk["id"] for chunk in result.chunks] == expected_ids
    assert result.confidence == plain.confidence


def test_a_failing_router_falls_back_to_the_whole_library(fresh_db, monkeypatch):
    library(fresh_db)
    question = "پاداش عملکرد چقدر است؟"
    expected_ids = [chunk["id"] for chunk in unrouted(question).chunks]
    stub_completion(monkeypatch, RuntimeError("provider is down"))
    result = run(search_module.search(settings(), question))
    assert result.routed_sections == []
    assert [chunk["id"] for chunk in result.chunks] == expected_ids


def test_an_unparseable_reply_falls_back_to_the_whole_library(fresh_db, monkeypatch):
    library(fresh_db)
    question = "سقف مرخصی استعلاجی چند روز است؟"
    expected_ids = [chunk["id"] for chunk in unrouted(question).chunks]
    stub_completion(monkeypatch, "I think it is probably in chapter two somewhere.")
    result = run(search_module.search(settings(), question))
    assert result.routed_sections == []
    assert [chunk["id"] for chunk in result.chunks] == expected_ids


def test_a_library_below_the_threshold_is_never_routed(fresh_db, monkeypatch):
    library(fresh_db)
    question = "سقف مرخصی استعلاجی چند روز است؟"
    expected_ids = [chunk["id"] for chunk in unrouted(question).chunks]
    calls = []
    stub_completion(monkeypatch, json.dumps({"sections": []}), calls)
    result = run(search_module.search(settings(toc_router_min_chunks=99), question))
    assert calls == [], "the router must not spend a model call below the threshold"
    assert [chunk["id"] for chunk in result.chunks] == expected_ids


def test_narrowing_does_not_deflate_confidence(fresh_db, monkeypatch):
    """Confidence is measured against the library, not the router's slice.

    Scoring the z-score over the narrowed pool made a single selected chunk
    look unremarkable by construction, and the evidence gate then refused a
    correct answer.
    """
    library(fresh_db)
    question = "سقف مرخصی استعلاجی چند روز است؟"
    stub_completion(monkeypatch, json.dumps({"sections": ["فصل دوم — مرخصی > مرخصی استعلاجی"]}))
    routed = run(search_module.search(settings(), question))
    assert routed.chunks, "narrowing to the right section must not cause an abstention"
    assert routed.evidence_found

    # The chunk's distinctiveness is a property of the library, so it must read
    # the same whether or not the router narrowed the search to it.
    plain = unrouted(question)
    selected = routed.chunks[0]
    same = next(chunk for chunk in plain.chunks if chunk["id"] == selected["id"])
    assert selected["standout"] == same["standout"]


def test_the_flag_being_off_leaves_the_old_path_untouched(fresh_db, monkeypatch):
    library(fresh_db)
    calls = []
    stub_completion(monkeypatch, json.dumps({"sections": []}), calls)
    routed = run(search_module.search(settings(toc_router_enabled=False), "سقف مرخصی استعلاجی چند روز است؟"))
    assert calls == []
    assert routed.routed_sections == []
    assert "route" not in routed.timings_ms
    assert routed.chunks


def test_routing_is_off_by_default():
    assert AppSettings().toc_router_enabled is False
