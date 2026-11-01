"""Pre-emptive context rotation for paperclip ``api_server`` issue sessions.

A per-issue paperclip session is long-lived by design: every heartbeat on one issue appends to the
same transcript, and the only thing that ever shrinks it is a compression pass. When that pass
fails — the provider truncates the summary at its output cap, the auxiliary lane is overloaded —
the transcript keeps growing until the model physically cannot read it, and the NEXT turn dies with
``context_overflow``. The heartbeat exits non-zero and the card keeps whatever roster label the dead
run left on it. A stranded heartbeat is not a model problem: it is a session that outlived its own
readability.

This module is the last gate before that wall. At turn start, when a paperclip issue-keyed session
is already close enough to the model's window that a compression pass WOULD be needed anyway, the
gateway rotates instead of summarising: the predecessor is closed with the ``context_rotation``
reset end-reason and the new transcript OPENS on the compaction summary the old transcript already
carries. The agent keeps working across the boundary — it just stops carrying a transcript the
model cannot read.

Scope is deliberately narrow: ``source=api_server`` sessions whose key is a paperclip conversation
(``paperclip:company:…:agent:…:issue:…``). Telegram/CLI/Discord transcripts are never touched —
those sessions are user-driven, and a rotation nobody asked for reads to a human as the bot
forgetting the conversation.

Config (under ``compression:``)::

    rotate_before_overflow: true     # default ON
    rotation_input_share: 0.85       # share of model context_length that triggers the rotation
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# End reason stamped on the closed predecessor — deliberately NOT in
# ``hermes_state_common._RESET_END_REASONS``. Registering it would make the child a legacy reset
# child, and ``resolve_resume_session_id`` skips those, so a heartbeat sending the old session id
# would resolve back to the closed parent. The reason stays greppable in state.db for whoever is
# attributing a traffic-cop hit.
CONTEXT_ROTATION_END_REASON = "context_rotation"

# The platform value the paperclip harness drives this gateway through, and the session-key prefix
# it mints per issue/agent conversation. BOTH must match: a chat-driven api_server caller (an
# operator scripting against /v1/runs) is out of scope, and a telegram turn can never qualify.
PAPERCLIP_SOURCE = "api_server"
PAPERCLIP_KEY_PREFIX = "paperclip:"

DEFAULT_ROTATION_ENABLED = True
DEFAULT_ROTATION_SHARE = 0.85
_MIN_SHARE = 0.5
_MAX_SHARE = 0.99
# A rotation that does not at least halve the transcript is not worth a session boundary; the
# caller falls through to the ordinary hygiene compression pass instead of burning a reset.
_MIN_SHRINK_RATIO = 0.5


def _as_bool(value: Any, default: bool) -> bool:
    """Loose bool for YAML values (bool, "true"/"on"/"1"); None keeps *default*."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def rotation_settings(compression_cfg: Any) -> Tuple[bool, float]:
    """``(enabled, share)`` from a ``compression:`` mapping. Invalid values keep the defaults."""
    if not isinstance(compression_cfg, dict):
        return DEFAULT_ROTATION_ENABLED, DEFAULT_ROTATION_SHARE
    enabled = _as_bool(compression_cfg.get("rotate_before_overflow"), DEFAULT_ROTATION_ENABLED)
    share = DEFAULT_ROTATION_SHARE
    raw = compression_cfg.get("rotation_input_share")
    if raw is not None:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0:
            # Clamped, not rejected: a 1.5 share would disable the gate silently, and a 0.2 share
            # would rotate a session that has barely started.
            share = min(_MAX_SHARE, max(_MIN_SHARE, parsed))
    return enabled, share


def is_paperclip_api_server_session(session_key: Optional[str], platform: Optional[str]) -> bool:
    """True only for a paperclip conversation arriving through the api_server adapter.

    This is the whole gate in one place: every rotation path calls it, so a telegram/discord/cli
    session can never reach the rotation even with ``rotate_before_overflow: true``.
    """
    if not session_key or not platform:
        return False
    if str(platform).strip().lower() != PAPERCLIP_SOURCE:
        return False
    return str(session_key).startswith(PAPERCLIP_KEY_PREFIX)


def _default_summary_finder(history: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    from agent.context_compressor import ContextCompressor

    start = 1 if history and history[0].get("role") == "system" else 0
    return ContextCompressor._find_context_summaries(history, start, len(history))


def _default_token_estimator(messages: Sequence[Dict[str, Any]]) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    return int(estimate_messages_tokens_rough(list(messages)))


@dataclass(frozen=True)
class ContextRotationDecision:
    """Outcome of the pre-turn rotation check.

    ``reason`` is always populated: ``"rotate"`` when ``rotate`` is True, else the gate that
    declined (logged at DEBUG so an operator can see why a near-wall session kept compressing).
    """

    rotate: bool
    reason: str
    projected_tokens: int = 0
    threshold_tokens: int = 0
    share: float = DEFAULT_ROTATION_SHARE
    context_length: int = 0
    summary_indices: Tuple[int, ...] = ()
    seeded_tokens: int = 0


def decide_context_rotation(
    *,
    enabled: bool,
    share: float,
    context_length: int,
    projected_tokens: int,
    session_key: Optional[str],
    platform: Optional[str],
    history: List[Dict[str, Any]],
    summary_finder: Optional[Callable[[List[Dict[str, Any]]], List[Tuple[int, str]]]] = None,
    token_estimator: Optional[Callable[[Sequence[Dict[str, Any]]], int]] = None,
) -> ContextRotationDecision:
    """Decide whether this turn should rotate the session instead of compressing it.

    Pure apart from two injectable probes (summary locator, token estimator), so the gate can be
    tested without a provider, a database, or a gateway.
    """
    share = min(_MAX_SHARE, max(_MIN_SHARE, float(share or DEFAULT_ROTATION_SHARE)))
    projected = max(0, int(projected_tokens or 0))
    window = max(0, int(context_length or 0))
    threshold = int(window * share) if window else 0
    base = dict(projected_tokens=projected, threshold_tokens=threshold, share=share, context_length=window)

    if not enabled:
        return ContextRotationDecision(False, "disabled", **base)
    if not is_paperclip_api_server_session(session_key, platform):
        return ContextRotationDecision(False, "not_a_paperclip_api_server_session", **base)
    if window <= 0:
        # No trustworthy window means no trustworthy wall: compressing is still safe, rotating is not.
        return ContextRotationDecision(False, "no_context_length", **base)
    if projected < threshold:
        return ContextRotationDecision(False, "under_share", **base)

    finder = summary_finder or _default_summary_finder
    try:
        hits = finder(history) or []
    except Exception:
        logger.debug("context rotation: summary scan failed", exc_info=True)
        hits = []
    if not hits:
        # Nothing has ever been compacted, so there is no handoff to open the new transcript with.
        # Rotating here would throw the whole conversation away — compress instead.
        return ContextRotationDecision(False, "no_compaction_summary", **base)

    indices = tuple(idx for idx, _body in hits)
    seeded = seeded_rotation_messages(history, indices)
    if not seeded:
        return ContextRotationDecision(False, "summary_rows_unusable", **base)
    estimate = token_estimator or _default_token_estimator
    try:
        seeded_tokens = int(estimate(seeded))
    except Exception:
        logger.debug("context rotation: seeded-size estimate failed", exc_info=True)
        seeded_tokens = 0
    if seeded_tokens >= projected * _MIN_SHRINK_RATIO:
        return ContextRotationDecision(
            False, "summary_not_smaller", summary_indices=indices, seeded_tokens=seeded_tokens, **base,
        )
    return ContextRotationDecision(
        True, "rotate", summary_indices=indices, seeded_tokens=seeded_tokens, **base,
    )


def seeded_rotation_messages(
    history: List[Dict[str, Any]], summary_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    """Copies of the compaction handoff rows — the opening context of the rotated transcript.

    Every handoff in the old transcript is carried, not just the newest: the compressor's own
    rehydration joins all of them into ``_previous_summary``, so dropping an older one here would
    silently shorten the agent's memory at the boundary.
    """
    from agent.context_compressor import _strip_persistence_markers

    rows: List[Dict[str, Any]] = []
    for idx in summary_indices or ():
        try:
            row = history[idx]
        except (IndexError, TypeError):
            continue
        if not isinstance(row, dict):
            continue
        content = row.get("content")
        if isinstance(content, str) and not content.strip():
            continue
        if content is None:
            continue
        copied = copy.deepcopy(row)
        copied.pop("id", None)
        rows.append(copied)
    if rows:
        # A leaked ``_db_persisted`` marker would make the child's flush skip the row, losing the
        # handoff from the very transcript that exists to carry it.
        _strip_persistence_markers(rows)
    return rows


def projected_input_tokens(
    *, db: Any = None, session_id: Optional[str] = None, history: Sequence[Dict[str, Any]],
) -> Tuple[int, str]:
    """``(tokens, source)`` for the conversation as it stands at turn start.

    The API-reported anchor from the last turn is the honest reading; the char-based estimate is
    the fallback, and it deliberately runs HIGH (a gate that fires early only costs a session
    boundary, a gate that fires late costs the heartbeat).
    """
    if db is not None and session_id:
        try:
            from agent.usage_anchor import persisted_anchor_tokens

            anchored = persisted_anchor_tokens(getattr(db, "_db", db), session_id, list(history))
            if anchored:
                return int(anchored), "anchored"
        except Exception:
            logger.debug("context rotation: usage-anchor read failed", exc_info=True)
    try:
        return _default_token_estimator(history), "estimated"
    except Exception:
        logger.debug("context rotation: token estimate failed", exc_info=True)
        return 0, "unavailable"


def publish_context_rotation(
    db: Any,
    *,
    parent_session_id: str,
    child_session_id: str,
    seed_messages: List[Dict[str, Any]],
    source: str = PAPERCLIP_SOURCE,
    model: Optional[str] = None,
) -> bool:
    """Atomically close *parent_session_id* and publish *child_session_id* seeded with the handoff.

    Reuses the compression-child publication — parent closure, child row and handoff in ONE
    transaction — with the rotation end-reason, so a concurrent reader sees either the live parent
    or a complete child, never an ended parent whose transcript is empty. The child inherits the
    parent's routing identity (``session_key`` included), which is what makes the paperclip
    harness's declared-conversation lookup resolve the continuation instead of minting a new id.

    Returns False with the parent untouched on any failure; the caller then falls through to the
    ordinary compression pass, which is the pre-existing behaviour.
    """
    if not parent_session_id or not child_session_id or not seed_messages:
        return False
    publish = getattr(db, "publish_compression_child", None)
    if not callable(publish):
        return False
    try:
        publish(
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            source=source,
            messages=seed_messages,
            model=model,
            end_reason=CONTEXT_ROTATION_END_REASON,
            require_compression_lease=False,
        )
    except Exception as e:
        logger.warning(
            "Context rotation %s -> %s could not be published (%s); keeping the parent session",
            parent_session_id, child_session_id, e,
        )
        return False
    return True


def context_rotation_log_line(
    *,
    old_session_id: str,
    new_session_id: str,
    session_key: str,
    message_count: int,
    projected_tokens: int,
    threshold_tokens: int,
    share: float,
    context_length: int,
    seeded_messages: int,
    seeded_tokens: int,
) -> str:
    """The one durable line a rotation leaves behind.

    Deliberately a single, stable, token-prefixed line: a traffic-cop hit on a paperclip agent can
    be attributed from the run log alone, without opening state.db.
    """
    return (
        f"paperclip-context-rotation session={old_session_id} -> {new_session_id} key={session_key} "
        f"msgs={int(message_count)} tokens={int(projected_tokens)} "
        f"threshold={int(threshold_tokens)} share={float(share):.2f} "
        f"context_length={int(context_length)} seeded_msgs={int(seeded_messages)} "
        f"seeded_tokens={int(seeded_tokens)} reason=before_overflow"
    )
