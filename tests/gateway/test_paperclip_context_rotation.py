"""Pre-emptive context rotation for paperclip issue sessions (CUS-3576).

Two things are pinned here:

* the GATE — only ``source=api_server`` sessions keyed by a paperclip conversation rotate, only
  when the transcript is already at a configurable share of the model's window AND a compaction
  summary exists to open the new transcript with. A telegram session never matches, at any share.
* the PUBLICATION — the predecessor is closed with ``end_reason='context_rotation'``, the child
  row inherits the routing key and carries the summary as its only message, the paperclip
  declared-conversation lookup resolves the child, and an unrelated telegram transcript is
  byte-identical before and after.
"""

import pytest

from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
from gateway.paperclip_rotation import (
    CONTEXT_ROTATION_END_REASON,
    context_rotation_log_line,
    decide_context_rotation,
    is_paperclip_api_server_session,
    publish_context_rotation,
    seeded_rotation_messages,
)
from hermes_state import SessionDB

PAPERCLIP_KEY = (
    "paperclip:company:14880aed-64a1-4b21-aa38-9209e6923834"
    ":agent:71d017b7-4556-4089-a7e7-4e44793455e2"
    ":issue:30e25782-5fa6-4fcc-9308-a708f96fea08"
)
TELEGRAM_KEY = "agent:telegram:dm:1968422"
PARENT = "20261101_002338_a1b2c3"
CHILD = "20261101_004512_d4e5f6"
CONTEXT_LENGTH = 32_000
PROJECTED = 40_000  # already past 0.85 * 32k: a compression pass would be needed anyway


@pytest.fixture()
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(
        PARENT, "api_server", session_key=PAPERCLIP_KEY, chat_id=PAPERCLIP_KEY,
        chat_type="dm", user_id="agent-71d017b7",
    )
    store.replace_messages(PARENT, [_summary_row()] + _tail(40))
    store.create_session(
        "tg-session", "telegram", session_key=TELEGRAM_KEY, chat_id="1968422",
        chat_type="dm", user_id="1968422",
    )
    store.replace_messages(
        "tg-session", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    yield store
    store.close()


def _summary_row(body="shipped the parser, the fixture and the LAND evidence for the previous card"):
    return {
        "role": "user",
        "content": f"{SUMMARY_PREFIX}\n## Historical Task Snapshot\n{body}\n\n{_SUMMARY_END_MARKER}",
        "_compressed_summary": True,
    }


def _tail(pairs):
    rows = []
    for i in range(pairs):
        rows.append({"role": "user", "content": f"step {i}: " + "u" * 400})
        rows.append({"role": "assistant", "content": f"done {i}: " + "a" * 400})
    return rows


def _decide(db, **overrides):
    history = db.get_messages_as_conversation(PARENT)
    kwargs = dict(
        enabled=True, share=0.85, context_length=CONTEXT_LENGTH, projected_tokens=PROJECTED,
        session_key=PAPERCLIP_KEY, platform="api_server", history=history,
    )
    kwargs.update(overrides)
    return decide_context_rotation(**kwargs), history


def test_gate_matches_only_paperclip_api_server_keys():
    assert is_paperclip_api_server_session(PAPERCLIP_KEY, "api_server")
    assert not is_paperclip_api_server_session(PAPERCLIP_KEY, "telegram")
    assert not is_paperclip_api_server_session(TELEGRAM_KEY, "api_server")
    assert not is_paperclip_api_server_session(TELEGRAM_KEY, "telegram")
    assert not is_paperclip_api_server_session("", "api_server")
    assert not is_paperclip_api_server_session(PAPERCLIP_KEY, None)


@pytest.mark.parametrize(
    "session_key,platform",
    [
        (TELEGRAM_KEY, "telegram"),
        (TELEGRAM_KEY, "api_server"),
        (PAPERCLIP_KEY, "telegram"),
        ("agent:api_server:dm:local-script", "api_server"),
    ],
)
def test_gate_never_rotates_outside_paperclip_api_server(db, session_key, platform):
    decision, _history = _decide(db, session_key=session_key, platform=platform)
    assert decision.rotate is False
    assert decision.reason == "not_a_paperclip_api_server_session"


def test_rotates_a_paperclip_session_at_the_share(db):
    decision, history = _decide(db)
    assert decision.rotate is True, decision.reason
    assert decision.reason == "rotate"
    assert decision.threshold_tokens == int(CONTEXT_LENGTH * 0.85)
    assert decision.summary_indices == (0,)
    assert decision.seeded_tokens < PROJECTED


def test_under_the_share_keeps_compressing(db):
    decision, _ = _decide(db, projected_tokens=10_000)
    assert decision.rotate is False
    assert decision.reason == "under_share"


def test_disabled_config_never_rotates(db):
    decision, _ = _decide(db, enabled=False)
    assert decision.rotate is False
    assert decision.reason == "disabled"


def test_no_compaction_summary_keeps_compressing(db):
    """Nothing was ever compacted: rotating would throw the conversation away, so compress instead."""
    db.replace_messages(PARENT, _tail(40))
    decision, _ = _decide(db)
    assert decision.rotate is False
    assert decision.reason == "no_compaction_summary"


def test_no_context_length_keeps_compressing(db):
    decision, _ = _decide(db, context_length=0)
    assert decision.rotate is False
    assert decision.reason == "no_context_length"


def test_publication_closes_parent_seeds_child_and_leaves_telegram_alone(db):
    decision, history = _decide(db)
    assert decision.rotate
    seeded = seeded_rotation_messages(history, decision.summary_indices)
    assert len(seeded) == 1
    assert "shipped the parser" in seeded[0]["content"]

    telegram_before = db.get_messages_as_conversation("tg-session")
    telegram_row_before = dict(db.get_session("tg-session"))

    assert publish_context_rotation(
        db, parent_session_id=PARENT, child_session_id=CHILD,
        seed_messages=seeded, model="test/model",
    )

    parent = dict(db.get_session(PARENT))
    assert parent["ended_at"] is not None
    assert parent["end_reason"] == CONTEXT_ROTATION_END_REASON == "context_rotation"

    child = dict(db.get_session(CHILD))
    assert child["ended_at"] is None
    assert child["parent_session_id"] == PARENT
    assert child["session_key"] == PAPERCLIP_KEY
    child_messages = db.get_messages_as_conversation(CHILD)
    assert len(child_messages) == 1
    assert "shipped the parser" in child_messages[0]["content"]

    # The paperclip harness resolves its conversation by declared key: it must land on the child,
    # never on the closed parent and never on a fresh empty id.
    live = db.find_latest_gateway_session_for_peer(source="api_server", session_key=PAPERCLIP_KEY)
    assert live is not None and live["id"] == CHILD
    # A heartbeat that still sends the OLD id must follow parent_session_id forward to the child:
    # context_rotation is not a reset boundary, so the child is not skipped as a legacy reset child.
    assert db.resolve_resume_session_id(PARENT) == CHILD

    # A telegram transcript is not a rotation candidate and must be untouched by one.
    assert db.get_messages_as_conversation("tg-session") == telegram_before
    assert dict(db.get_session("tg-session"))["ended_at"] == telegram_row_before["ended_at"]
    assert dict(db.get_session("tg-session"))["end_reason"] == telegram_row_before["end_reason"]


def test_publication_refuses_an_empty_seed(db):
    """An empty seed would publish a child with no opening context; the parent must stay live."""
    assert publish_context_rotation(
        db, parent_session_id=PARENT, child_session_id=CHILD, seed_messages=[],
    ) is False
    parent = dict(db.get_session(PARENT))
    assert parent["ended_at"] is None
    assert db.get_session(CHILD) is None


def test_log_line_names_the_attribution_fields():
    line = context_rotation_log_line(
        old_session_id=PARENT, new_session_id=CHILD, session_key=PAPERCLIP_KEY,
        message_count=165, projected_tokens=579_853, threshold_tokens=544_768,
        share=0.85, context_length=640_903, seeded_messages=2, seeded_tokens=4_100,
    )
    assert line.startswith("paperclip-context-rotation ")
    for needle in (
        f"session={PARENT}", f"-> {CHILD}", "msgs=165", "tokens=579853",
        "threshold=544768", "share=0.85", "context_length=640903", "reason=before_overflow",
    ):
        assert needle in line
    assert "\n" not in line
