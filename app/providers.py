from __future__ import annotations

import re
import logging
from typing import Any

import httpx

from .claims import ABSTAIN_MARKER
from .models import AppSettings
from .net import ensure_success, post_with_retry


logger = logging.getLogger("adaptive_metric_rag.providers")


def _has_complete_ending(answer: str) -> bool:
    clean = answer.strip()
    if not clean:
        return False
    # A citation or closing Markdown delimiter may follow the real punctuation.
    tail = re.sub(r"(?:\s*\[\d+\])+\s*$", "", clean)
    tail = re.sub(r"(?:\*\*|__|`)+\s*$", "", tail).rstrip()
    return tail.endswith((".", "!", "?", "؟", "؛", "…", ")", "]", "}", "»", '"', "'"))


def _looks_incomplete(question: str, answer: str, finish_reason: str = "",
                      strict_multipart: bool = False) -> bool:
    """Whether the answer was cut off, as opposed to merely being short.

    Length alone used to count: a two-part question whose answer ran under 350
    characters was sent back for repair. Two dates and two amounts are a
    complete answer to a two-part question, so the rule was rewriting correct
    answers and paying for a second call to do it. It is now opt-in through
    `strict_multipart_answers`; the structural checks below stand on their own.
    """
    clean = answer.strip()
    if not clean:
        return True
    if ABSTAIN_MARKER in clean:
        return False  # refusing is a complete answer, however short
    if finish_reason.lower() in {"length", "max_tokens", "max_tokens_reached"}:
        return True
    unbalanced = clean.count("(") > clean.count(")") or clean.count("**") % 2 == 1
    broken_ending = clean.endswith(("(", "[", "{", ":", "،", ",", "-", "**"))
    multi_part = question.count("؟") + question.count("?") >= 2
    missing_sentence_end = not _has_complete_ending(clean)
    return (unbalanced or broken_ending or missing_sentence_end
            or (strict_multipart and multi_part and len(clean) < 350))


def _repair_instruction(language: str) -> str:
    if language == "fa":
        return ("پاسخ قبلی ناقص، قطع‌شده یا به زبان اشتباه بود. یک پاسخ کامل جایگزین بنویس؛ "
                "تمام متن توضیحی را فقط به فارسی بنویس، همه بخش‌های سؤال را جداگانه پاسخ بده، "
                "چیزی را تکرار نکن و ارجاع‌های [1]، [2] را حفظ کن. حداکثر ۱۸۰ کلمه بنویس و حتماً با یک جمله کامل تمام کن. "
                "نام خاص، ISBN و URL می‌توانند لاتین بمانند.")
    return "The previous answer was incomplete or cut off. Write a complete replacement in at most 180 words, answer every part separately, avoid repetition, preserve [1], [2] citations, and end with a complete sentence."


def _continuation_instruction(language: str) -> str:
    if language == "fa":
        return ("پاسخ دقیقاً در انتهای متن قطع شده است. فقط از همان عبارت ناتمام ادامه بده؛ متن قبلی را تکرار نکن، "
                "پاسخ را در حداکثر ۱۲۰ کلمه جمع‌بندی کن، ارجاع‌ها را حفظ کن و حتماً با جمله کامل پایان بده.")
    return ("The answer was cut off at the very end. Continue exactly from the unfinished phrase without repeating "
            "the previous text, finish within 120 words, preserve citations, and end with a complete sentence.")


def _join_continuation(answer: str, continuation: str) -> str:
    return answer.rstrip() + ("" if answer.rstrip().endswith(("-", "—")) else " ") + continuation.lstrip()


def _trim_incomplete_tail(answer: str) -> str:
    """Remove only a final dangling fragment when a provider repeatedly stops early."""
    clean = answer.strip()
    if _has_complete_ending(clean):
        return clean
    endings = list(re.finditer(r"[.!?؟؛…](?:\s*\[\d+\])*", clean))
    if endings and endings[-1].end() >= 40:
        return clean[:endings[-1].end()].rstrip()
    return clean


def _language_instruction(language: str) -> str:
    if language == "fa":
        return ("قانون الزامی زبان: پاسخ را فقط به فارسی روان بنویس. متن انگلیسی منابع را ترجمه و به فارسی توضیح بده. "
                "فقط نام‌های خاص، کد، ISBN و نشانی اینترنتی را در صورت نیاز لاتین نگه دار. این قانون بر همه دستورهای دیگر مقدم است.")
    return "Mandatory language rule: write the complete answer in English."


def _abstain_instruction(language: str) -> str:
    """One exact token for "the sources do not answer this".

    Detecting abstention by matching prose meant keeping a list of the phrases
    each model happens to use, in two languages, and being quietly wrong about
    the next one. A token the model is told to emit is checked exactly.
    """
    if language == "fa":
        return (f"اگر منابع بالا پاسخ سؤال را ندارند، دقیقاً همین نشانه را بنویس: {ABSTAIN_MARKER} "
                "و هیچ پاسخ حدسی ننویس.")
    return (f"If the sources above do not answer the question, reply with exactly this token: "
            f"{ABSTAIN_MARKER} - and do not guess.")


def _structured_instruction(language: str, count: int) -> str:
    """Ask for the answer and its claims as JSON, when the caller wants them."""
    shape = ('{"answer": "<the full answer text>", '
             '"claims": [{"text": "<one sentence of the answer>", "source_ids": [1]}]}')
    if language == "fa":
        return (f"خروجی را فقط به شکل JSON بنویس: {shape} — هر جمله پاسخ یک claim است و "
                f"source_ids شماره منابع (۱ تا {count}) است که همان جمله بر آن‌ها تکیه دارد. "
                "جمله‌ای که به هیچ منبعی تکیه ندارد source_ids خالی می‌گیرد.")
    return (f"Reply with JSON only, in this shape: {shape} - one claim per sentence of the answer, "
            f"source_ids being the source numbers (1 to {count}) that sentence rests on. "
            "A sentence resting on no source gets an empty source_ids.")


def _wrong_language(language: str, answer: str) -> bool:
    """Detect a clearly English response to a Persian question without rejecting names/URLs."""
    if language != "fa":
        return False
    prose = re.sub(r"https?://\S+|\[[0-9]+\]|`[^`]*`", " ", answer)
    persian = len(re.findall(r"[\u0600-\u06ff]", prose))
    latin = len(re.findall(r"[A-Za-z]", prose))
    return latin >= 35 and (persian < 18 or persian < latin * .22)


def _finalize_answer(answer: str) -> str:
    """Prevent a repaired model response from ending with open formatting."""
    clean = _trim_incomplete_tail(answer)
    if clean.count("**") % 2 == 1:
        clean += "**"
    missing_parentheses = clean.count("(") - clean.count(")")
    if 0 < missing_parentheses <= 3:
        clean += ")" * missing_parentheses
    missing_brackets = clean.count("[") - clean.count("]")
    if 0 < missing_brackets <= 3:
        clean += "]" * missing_brackets
    if clean.endswith((":", "،", ",", "-")):
        clean = clean.rstrip(":،,- ") + "."
    return clean


def _history_block(history: list[dict[str, str]] | None, language: str) -> str:
    """The recent turns, so a follow-up reads as a follow-up.

    Retrieval already resolved the question against the conversation; the model
    needs the same context to write «مبلغ همان قرارداد ...» instead of asking
    which contract is meant. The sources still decide the facts: the history is
    labelled as background, never as material to answer from.
    """
    turns = [turn for turn in (history or []) if turn.get("content")]
    if not turns:
        return ""
    lines = "\n".join(f"{turn.get('role', 'user')}: {turn['content'].strip()}" for turn in turns)
    if language == "fa":
        return ("\n\nگفت‌وگوی پیشین (فقط برای فهم منظور سؤال؛ پاسخ باید تنها از منابع زیر بیاید):\n"
                f"{lines}")
    return ("\n\nEarlier conversation (context for understanding the question only; "
            f"answer strictly from the sources below):\n{lines}")


def build_prompt(question: str, chunks: list[dict[str, Any]], system_prompt: str, language: str,
                 history: list[dict[str, str]] | None = None, structured: bool = False) -> tuple[str, str]:
    sources = "\n\n".join(
        f"[{i}] Document: {chunk['document_name']} | Page: {chunk.get('page') or '-'}\n{chunk['content']}"
        for i, chunk in enumerate(chunks, 1)
    )
    context = _history_block(history, language)
    if language == "fa":
        user = (f"سؤال:\n{question}{context}\n\nمنابع:\n{sources}\n\n"
                "پاسخی کامل، مستقیم و مستند به همین منابع ارائه کن. تمام بخش‌های سؤال را پاسخ بده، "
                "ارجاع درون‌متنی [1]، [2] درج کن و جمله را نیمه‌تمام رها نکن. پاسخ باید فارسی باشد.")
    else:
        user = (f"Question:\n{question}{context}\n\nSources:\n{sources}\n\nGive a complete, direct answer "
                "grounded in these sources. Answer every requested part separately, include inline citations, "
                "and do not stop mid-sentence.")
    user += "\n\n" + _abstain_instruction(language)
    if structured:
        user += "\n\n" + _structured_instruction(language, len(chunks))
    return system_prompt.strip() + "\n\n" + _language_instruction(language), user


def local_answer(question: str, chunks: list[dict[str, Any]], language: str) -> str:
    if not chunks:
        return "سند مرتبطی پیدا نشد. لطفاً منبع مناسب اضافه کنید." if language == "fa" else "No relevant source was found. Please add a suitable document."
    excerpts = []
    for i, chunk in enumerate(chunks[:3], 1):
        sentences = re.split(r"(?<=[.!?؟])\s+|\n+", chunk["content"])
        excerpt = next((s.strip() for s in sentences if len(s.strip()) > 35), chunk["content"][:320]).strip()
        excerpts.append(f"{excerpt} [{i}]")
    prefix = "بر اساس منابع بازیابی‌شده:" if language == "fa" else "Based on the retrieved sources:"
    return prefix + "\n\n" + "\n\n".join(excerpts)


async def generate(settings: AppSettings, question: str, chunks: list[dict[str, Any]], language: str,
                   history: list[dict[str, str]] | None = None, structured: bool = False) -> str:
    if settings.provider == "local":
        return local_answer(question, chunks, language)
    system, user = build_prompt(question, chunks, settings.system_prompt, language, history,
                                structured=structured)
    strict = settings.strict_multipart_answers
    timeout = httpx.Timeout(150.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if settings.provider == "ollama":
            base = (settings.base_url or "http://host.docker.internal:11434").rstrip("/")
            response = await post_with_retry(client, f"{base}/api/chat", what="Ollama generation", json={
                "model": settings.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": settings.temperature, "num_predict": settings.max_tokens},
            })
            ensure_success(response, "generation")
            payload = response.json()
            answer = payload["message"]["content"]
            done_reason = payload.get("done_reason", "")
            logger.info("Ollama generation finished: reason=%s chars=%d eval_count=%s complete=%s", done_reason, len(answer), payload.get("eval_count"), not _looks_incomplete(question, answer, done_reason, strict))
            if _wrong_language(language, answer):
                repair = await post_with_retry(client, f"{base}/api/chat", what="Ollama repair", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer},
                                 {"role": "user", "content": _repair_instruction(language)}],
                    "options": {"temperature": min(settings.temperature, .2), "num_predict": settings.max_tokens},
                })
                ensure_success(repair, "repair")
                repaired_payload = repair.json()
                answer = repaired_payload["message"]["content"]
                done_reason = repaired_payload.get("done_reason", "")
                logger.info("Ollama language repair finished: reason=%s chars=%d eval_count=%s complete=%s", done_reason, len(answer), repaired_payload.get("eval_count"), not _looks_incomplete(question, answer, done_reason, strict))
            for attempt in range(2):
                if not _looks_incomplete(question, answer, done_reason, strict):
                    break
                continuation = await post_with_retry(client, f"{base}/api/chat", what="Ollama continuation", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer},
                                 {"role": "user", "content": _continuation_instruction(language)}],
                    "options": {"temperature": min(settings.temperature, .2), "num_predict": min(settings.max_tokens, 512)},
                })
                ensure_success(continuation, "continuation")
                continuation_payload = continuation.json()
                piece = continuation_payload["message"]["content"]
                answer = _join_continuation(answer, piece)
                done_reason = continuation_payload.get("done_reason", "")
                logger.info("Ollama continuation %d finished: reason=%s piece_chars=%d total_chars=%d complete=%s", attempt + 1, done_reason, len(piece), len(answer), not _looks_incomplete(question, answer, done_reason, strict))
            return _finalize_answer(answer)
        if settings.provider == "openai":
            base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
            response = await post_with_retry(client, f"{base}/chat/completions", what="OpenAI generation", headers={"Authorization": f"Bearer {settings.api_key}"}, json={
                "model": settings.model, "temperature": settings.temperature, "max_tokens": settings.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
            ensure_success(response, "generation")
            payload = response.json()
            choice = payload["choices"][0]
            answer = choice["message"]["content"]
            if _looks_incomplete(question, answer, choice.get("finish_reason", ""), strict) or _wrong_language(language, answer):
                repair = await post_with_retry(client, f"{base}/chat/completions", what="OpenAI repair", headers={"Authorization": f"Bearer {settings.api_key}"}, json={
                    "model": settings.model, "temperature": min(settings.temperature, .2), "max_tokens": settings.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": answer}, {"role": "user", "content": _repair_instruction(language)}],
                })
                ensure_success(repair, "repair")
                answer = repair.json()["choices"][0]["message"]["content"]
            return _finalize_answer(answer)
        if settings.provider == "gemini":
            base = (settings.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            response = await post_with_retry(client, f"{base}/models/{settings.model}:generateContent", what="Gemini generation", params={"key": settings.api_key}, json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": settings.temperature, "maxOutputTokens": settings.max_tokens},
            })
            ensure_success(response, "generation")
            payload = response.json()
            candidate = payload["candidates"][0]
            answer = candidate["content"]["parts"][0]["text"]
            if _looks_incomplete(question, answer, candidate.get("finishReason", ""), strict) or _wrong_language(language, answer):
                repair_prompt = user + "\n\n" + _repair_instruction(language)
                repair = await post_with_retry(client, f"{base}/models/{settings.model}:generateContent", what="Gemini repair", params={"key": settings.api_key}, json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": repair_prompt}]}],
                    "generationConfig": {"temperature": min(settings.temperature, .2), "maxOutputTokens": settings.max_tokens},
                })
                ensure_success(repair, "repair")
                answer = repair.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _finalize_answer(answer)
    raise ValueError(f"Unknown provider: {settings.provider}")


async def complete(settings: AppSettings, system: str, user: str, model: str = "",
                   temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """One plain completion against the configured provider.

    Used by stages that need the model to do something other than answer the
    user, such as scoring a shortlist for reranking. Kept separate from
    `generate` so the answer path keeps its repair and continuation behaviour
    and this one stays predictable.
    """
    model = model or settings.model
    timeout = httpx.Timeout(90.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if settings.provider == "ollama":
            base = (settings.base_url or "http://host.docker.internal:11434").rstrip("/")
            response = await post_with_retry(client, f"{base}/api/chat", what="Ollama completion", json={
                "model": model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_predict": max_tokens},
            })
            ensure_success(response, "Ollama completion")
            return response.json()["message"]["content"]
        if settings.provider == "openai":
            base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
            response = await post_with_retry(
                client, f"{base}/chat/completions",
                what="OpenAI completion",
                headers={"Authorization": f"Bearer {settings.api_key}"},
                json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
            )
            ensure_success(response, "OpenAI completion")
            return response.json()["choices"][0]["message"]["content"]
        if settings.provider == "gemini":
            base = (settings.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            response = await post_with_retry(
                client, f"{base}/models/{model}:generateContent",
                what=f"Gemini completion with model '{model}'",
                params={"key": settings.api_key},
                json={"system_instruction": {"parts": [{"text": system}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}},
            )
            ensure_success(response, f"Gemini completion with model '{model}'")
            parts = response.json()["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
    raise ValueError(f"Provider {settings.provider} cannot run a plain completion")

