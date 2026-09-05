"""Phase 6: streaming, structured citations, and checking what was claimed."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.claims import ABSTAIN_MARKER, Claim, claims_from_markers, parse_structured, with_markers
from app.main import answer_abstained, claim_for_citation
from app.models import AppSettings
from app.providers import _looks_incomplete, build_prompt
from app.streaming import _gemini_delta, _ollama_delta, _openai_delta, sse, stream_answer
from app.verify import check_claim, support_score, unsupported_figures, verify

from tests.test_api_smoke import CONTRACT, RUNBOOK, upload


# --- abstention is a token now, not a phrase we hope to recognise ---


def test_the_abstain_token_is_detected_in_either_language():
    assert answer_abstained(f"  {ABSTAIN_MARKER}\n")
    assert answer_abstained(f"متأسفانه {ABSTAIN_MARKER}")
    assert not answer_abstained("مبلغ کل قرارداد ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است [1].")


def test_answers_written_before_the_token_existed_are_still_recognised():
    assert answer_abstained("پاسخ این سؤال در منابع موجود پیدا نشد.")
    assert answer_abstained("The answer was not found in the available sources.")


def test_the_prompt_asks_for_the_token():
    _, user = build_prompt("چقدر؟", [{"document_name": "d", "content": "c"}], "sys", "fa")
    assert ABSTAIN_MARKER in user


def test_a_bare_abstention_is_not_treated_as_a_truncated_answer():
    assert not _looks_incomplete("چقدر؟", ABSTAIN_MARKER)


# --- claims ---


def test_markers_become_one_claim_per_sentence():
    claims = claims_from_markers("The fee is EUR 24,000 [2]. It is payable in thirty days [2][3]. "
                                 "Nothing else is stated.")
    assert [claim.source_ids for claim in claims] == [[2], [2, 3], []]
    assert claims[0].text.startswith("The fee")


def test_structured_output_is_read_when_the_model_returns_it():
    raw = json.dumps({"answer": "The fee is EUR 24,000.",
                      "claims": [{"text": "The fee is EUR 24,000.", "source_ids": [1]}]})
    answer, claims = parse_structured(raw, source_count=2)
    assert answer == "The fee is EUR 24,000."
    assert claims[0].source_ids == [1]


def test_a_claim_citing_a_source_that_was_never_supplied_loses_that_citation():
    raw = json.dumps({"answer": "A.", "claims": [{"text": "A.", "source_ids": [1, 9]}]})
    _, claims = parse_structured(raw, source_count=2)
    assert claims[0].source_ids == [1]


@pytest.mark.parametrize("raw", ["", "not json", "{}", '{"answer": ""}', '{"claims": []}'])
def test_unusable_structured_output_is_refused_so_the_markers_are_used(raw):
    assert parse_structured(raw, source_count=2) is None


def test_a_structured_answer_is_rendered_back_into_inline_markers():
    claims = [Claim(text="The fee is EUR 24,000.", source_ids=[1]),
              Claim(text="It is invoiced quarterly.", source_ids=[2])]
    rendered = with_markers("The fee is EUR 24,000. It is invoiced quarterly.", claims)
    assert rendered == "The fee is EUR 24,000 [1]. It is invoiced quarterly [2]."


def test_an_answer_that_already_has_markers_is_left_alone():
    text = "The fee is EUR 24,000 [1]."
    assert with_markers(text, [Claim(text=text, source_ids=[1])]) == text


# --- verification ---


def test_a_number_the_source_does_not_contain_is_unsupported():
    source = "The annual subscription fee is EUR 186,000, invoiced quarterly."
    assert unsupported_figures("The annual fee is EUR 186,000.", source) == []
    assert unsupported_figures("The annual fee is EUR 900,000.", source) == ["900000"]


def test_a_claim_about_another_topic_is_unsupported_even_without_numbers():
    source = "Northwind Analytics guarantees 99.9 percent monthly availability."
    assert check_claim(Claim(text="The onboarding covers fourteen warehouse locations.",
                             source_ids=[1]), {1: source}) is False


def test_a_claim_the_source_states_is_supported():
    source = "A one-time onboarding fee of EUR 24,000 is payable within thirty days."
    assert check_claim(Claim(text="The onboarding fee is EUR 24,000.", source_ids=[1]), {1: source})


def test_a_claim_with_no_citation_is_not_judged():
    assert check_claim(Claim(text="Generally speaking, contracts vary."), {1: "anything"}) is None


def test_a_citation_pointing_at_nothing_is_unsupported():
    assert check_claim(Claim(text="Something.", source_ids=[4]), {1: "text"}) is False


def test_support_score_is_a_share_of_the_claim_not_of_the_source():
    assert support_score("blocked condenser coil", "the blocked condenser coil was flagged") == 1.0
    assert support_score("entirely different words here", "the blocked condenser coil") < .3


def test_verification_can_be_switched_off():
    claims = [Claim(text="Anything at all.", source_ids=[1])]
    checked = asyncio.run(verify(AppSettings(verify_claims=False), claims, {1: "unrelated"}))
    assert checked[0].supported is None


def test_the_lexical_check_runs_by_default():
    claims = [Claim(text="The fee is EUR 900,000.", source_ids=[1])]
    checked = asyncio.run(verify(AppSettings(), claims, {1: "The fee is EUR 24,000."}))
    assert checked[0].supported is False


# --- the highlight for a citation the model did not mark ---


def test_an_unmarked_citation_gets_the_sentence_closest_to_its_source():
    answer = "The contract runs for twelve months. The total amount is 2,400,000,000 rial."
    claim = claim_for_citation(answer, 1, "The total amount of the contract is 2,400,000,000 rial.")
    assert claim == "The total amount is 2,400,000,000 rial."


def test_a_marked_citation_still_wins():
    answer = "First sentence [1]. Second sentence."
    assert claim_for_citation(answer, 1, "second sentence") == "First sentence [1]."


# --- streaming ---


def test_each_provider_shape_is_read_into_the_same_delta():
    assert _ollama_delta(json.dumps({"message": {"content": "hi"}, "done": False})) == ("hi", None)
    assert _openai_delta(json.dumps({"choices": [{"delta": {"content": "hi"}}]})) == ("hi", None)
    assert _gemini_delta(json.dumps({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})) == ("hi", None)


def test_a_finished_stream_reports_why_it_stopped():
    assert _ollama_delta(json.dumps({"message": {"content": ""}, "done": True,
                                     "done_reason": "length"}))[1] == "length"
    assert _openai_delta(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))[1] == "stop"


def test_an_event_survives_a_newline_in_its_payload():
    """A raw newline would end the event early and truncate the answer."""
    rendered = sse("delta", {"text": "first\nsecond"})
    assert rendered.count("\n\n") == 1 and rendered.endswith("\n\n")
    assert json.loads(rendered.splitlines()[1][5:].strip())["text"] == "first\nsecond"


def test_the_local_provider_streams_the_same_answer_it_returns():
    chunks = [{"document_name": "d.txt", "content": "The total amount is 2,400,000,000 rial. " * 3}]
    pieces = []

    async def collect():
        async for kind, piece in stream_answer(AppSettings(provider="local"), "how much?", chunks, "en"):
            pieces.append((kind, piece))

    asyncio.run(collect())
    assert pieces[-1][0] == "done"
    assert "".join(piece for kind, piece in pieces if kind == "delta") == pieces[-1][1]


# --- the endpoint ---


@pytest.fixture
def client(fresh_db):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def read_events(response) -> list[tuple[str, dict]]:
    events, name = [], ""
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[7:].strip()
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_streaming_sends_the_sources_before_the_answer(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    response = client.post("/api/chat/stream", json={"message": "مبلغ کل قرارداد چقدر است؟"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = read_events(response)
    assert events[0][0] == "meta"
    assert events[0][1]["analysis"]["language"] == "fa"
    assert [name for name, _ in events].count("done") == 1

    final = events[-1][1]
    streamed = "".join(payload["text"] for name, payload in events if name == "delta")
    assert final["answer"].startswith(streamed[:40])
    assert final["citations"][0]["document_name"] == "contract-137.fa.txt"


def test_a_streamed_answer_is_stored_like_any_other(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    response = client.post("/api/chat/stream", json={"message": "مبلغ کل قرارداد چقدر است؟"})
    conversation_id = read_events(response)[-1][1]["conversation_id"]

    messages = client.get(f"/api/conversations/{conversation_id}").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert json.loads(messages[1]["citations"])


def test_streaming_abstains_on_a_corpus_that_cannot_answer(client):
    upload(client, "gpu-runbook.en.md", RUNBOOK)
    response = client.post("/api/chat/stream",
                           json={"message": "قرارداد شماره ۱۳۷ چه زمانی تمام می‌شود؟"})
    final = read_events(response)[-1][1]
    assert final["evidence_found"] is False
    assert final["citations"] == []


def test_the_chat_response_carries_verified_claims(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    body = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert body["claims"], body
    cited = [claim for claim in body["claims"] if claim["source_ids"]]
    assert cited and all(claim["supported"] is True for claim in cited), cited
