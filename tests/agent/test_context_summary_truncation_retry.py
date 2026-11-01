"""A summary cut off by the output cap must be recoverable, not fatal (CUS-3576).

The 2026-11-01 incident: a paperclip issue session's compression pass hit
``finish_reason=length``, ``_generate_summary`` raised, the pass aborted with the transcript
unchanged, and the NEXT turn died with ``context_overflow`` — a dead heartbeat off a
compressible conversation. These tests pin the recovery: one retry against the newest part of
the window, an explicit note that the dropped turns are not covered, and the old fail-closed
behaviour preserved for everything else.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _compressor(**overrides):
    kwargs = dict(model="test/model", quiet_mode=True, tail_mode="legacy")
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(**kwargs)


def _turns(count=8):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} of the issue thread"}
        for i in range(count)
    ]


def _length_response():
    return {"choices": [{"finish_reason": "length", "message": {"content": "partial summary that stopped"}}]}


def _stop_response(text="full summary of the compacted turns"):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


def test_truncated_summary_retries_on_a_smaller_window():
    """First call hits the output cap; the retry on the newest half succeeds and the pass returns it."""
    responses = [_length_response(), _stop_response()]
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        out = _compressor()._generate_summary(_turns(8))

    assert out is not None, "a truncated summary must not kill the compression pass"
    assert "full summary of the compacted turns" in out
    assert out.startswith(SUMMARY_PREFIX)
    assert len(calls) == 2, "exactly one retry"
    # The retry summarises a NARROWER window and says so, instead of pretending full coverage.
    assert len(calls[1]["messages"][0]["content"]) > 0
    assert "were dropped before summarising" in calls[1]["messages"][0]["content"]


def test_second_truncation_still_fails_closed():
    """One retry only: a truncated retry keeps the existing abort path (no partial checkpoint)."""
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _length_response()

    compressor = _compressor()
    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        out = compressor._generate_summary(_turns(8))

    assert out is None
    assert len(calls) == 2
    assert getattr(compressor, "_last_summary_truncated_failure", False) is True


def test_small_window_is_not_shrunk():
    """A single exchange has nothing to drop: the pre-existing single-attempt failure stands."""
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _length_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        out = _compressor()._generate_summary(_turns(2))

    assert out is None
    assert len(calls) == 1


def test_non_truncation_failures_are_untouched():
    """An empty-content failure keeps its own path (main-model fallback / cooldown), never the retry."""
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]}

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        out = _compressor()._generate_summary(_turns(8))

    assert out is None
    assert len(calls) == 1, "the shrink retry is for finish_reason=length only"
