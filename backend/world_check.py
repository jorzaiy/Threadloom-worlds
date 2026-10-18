#!/usr/bin/env python3
"""World Check: a per-turn Nemotron call that tests the player's action
against the active ``world-bible.json``.

It runs beside the keyword arbiter (``arbiter_runtime.run_arbiter``) and
returns one of three verdicts:

* ``allowed`` – the world permits this and nothing notable follows.
* ``allowed_with_cost`` – the world permits the attempt, but it is illegal,
  taboo, dangerous, scarce or socially costly; the narrator must make the
  consequence land.
* ``out_of_world`` – the action needs technology, magic, knowledge or
  institutions that do not exist in this world (anachronisms, powers the
  bible rules out). The narrator must not render it as real.

Breaking a law or a taboo is *never* ``out_of_world``: the world can punish
a crime, it cannot punish a smartphone that does not exist.

The check is advisory and fail-open. Missing bible, missing NEBIUS_API_KEY,
timeouts, malformed JSON – all return ``ran=False`` (or a ``skipped`` status)
and the turn continues exactly as it would without World Check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from .character_assets import character_source_base
    from .llm_manager import get_role_runtime
    from .local_model_client import parse_json_response
    from .model_client import call_model
    from .model_config import nebius_api_key_configured, resolve_provider_model
except ImportError:  # flat-script imports (backend/ on sys.path)
    from character_assets import character_source_base
    from llm_manager import get_role_runtime
    from local_model_client import parse_json_response
    from model_client import call_model
    from model_config import nebius_api_key_configured, resolve_provider_model

logger = logging.getLogger(__name__)

WORLD_CHECK_ROLE = 'world_check'
WORLD_BIBLE_FILE = 'world-bible.json'
ENABLE_ENV = 'THREADLOOM_WORLD_CHECK'

VERDICT_ALLOWED = 'allowed'
VERDICT_COST = 'allowed_with_cost'
VERDICT_OUT = 'out_of_world'
VERDICTS = (VERDICT_ALLOWED, VERDICT_COST, VERDICT_OUT)

BIBLE_CHAR_LIMIT = 4000
ACTION_CHAR_LIMIT = 800
SCENE_CHAR_LIMIT = 400
FIELD_CHAR_LIMIT = 240

WORLD_CHECK_SYSTEM = """You are the World Check for a roleplay engine. You read a world bible and the
player's intended action, and decide whether THIS WORLD permits the attempt.
You do not judge whether the attempt succeeds, whether it is wise, or whether
it is moral. Card rules ([CARD]) are canon; [INFERRED] rules are softer.

Verdicts:
- "allowed": possible in this world, and nothing in the bible makes it costly.
- "allowed_with_cost": possible in this world, but the bible makes it illegal,
  taboo, dangerous, scarce, expensive, or socially punished. Crimes, sacrilege,
  breaking a curfew, handling the dead, defying a faction: ALL of these are
  allowed_with_cost, never out_of_world. Give the concrete in-world price.
- "out_of_world": the action needs technology, magic, creatures, knowledge or
  institutions that do not exist in this world (an anachronism, a power the
  bible rules out or limits away). Only physical/metaphysical impossibility
  belongs here.

Rules of thumb:
- If you can imagine a character in this world trying it and being punished
  for it, it is allowed_with_cost.
- If the world itself has no such thing to try with, it is out_of_world.
- Words, emotions, ordinary movement and everyday objects are allowed unless
  the bible says otherwise.
- "rule": first find the ONE bible line that is actually about this action
  (the line that mentions the object, power, place or act in question) and
  copy its key clause. A rule about firearms never decides a phone or a
  spell. Empty string if no line applies.
- "narrator_hint": one sentence, written in the SAME LANGUAGE as the player
  action (Chinese action -> Chinese hint), telling the narrator how to keep
  the turn inside the world. Do not decide whether the attempt succeeds and
  do not invent outcomes; say what the world makes visible: who notices,
  what the price looks like. For out_of_world, the impossible thing must
  never appear as real (no phone that "dies", no spell that "fizzles");
  name the concrete in-world substitute for the intent (e.g. asking the
  ferryman for the schedule instead of a phone).

Reply ONLY with a JSON object, keys in this order:
{"rule": "...",
 "verdict": "allowed" | "allowed_with_cost" | "out_of_world",
 "cost": "concrete in-world cost if allowed_with_cost, else \\"\\"",
 "narrator_hint": "..."}"""

_SKIP_PATTERNS = (
    re.compile(r'^\s*[/!#]'),            # slash / bang commands
    re.compile(r'^\s*[\(（\[【]\s*ooc', re.IGNORECASE),  # explicit OOC
)


def world_check_enabled() -> bool:
    value = str(os.environ.get(ENABLE_ENV, '1') or '').strip().lower()
    return value not in ('0', 'false', 'off', 'no')


def load_world_bible(source_root: Path | None = None) -> dict:
    root = source_root or character_source_base()
    path = Path(root) / WORLD_BIBLE_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as err:
        logger.warning('world_check: cannot read %s: %s', path, err)
        return {}
    return payload if isinstance(payload, dict) else {}


def is_canon_entry(entry: dict, *, locked: bool) -> bool:
    """Card text and the user's own additions are always canon. Once the user
    locks the bible, everything in it (Nemotron entries included) is canon:
    locking is the user accepting the world as written."""
    return locked or entry.get('source') in ('card', 'user')


def compact_bible(bible: dict, limit: int = BIBLE_CHAR_LIMIT) -> str:
    """Card canon first (it must never be cut), then inferred detail."""
    card: list[str] = []
    inferred: list[str] = []
    sections = bible.get('sections') if isinstance(bible.get('sections'), dict) else {}
    locked = bible.get('locked') is True
    for section, entries in sections.items():
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            text = ' '.join(str(entry.get('text', '') or '').split())
            if not text:
                continue
            is_card = is_canon_entry(entry, locked=locked)
            region = ' '.join(str(entry.get('region', '') or '').split())
            prefix = f'({region}) ' if region else ''
            line = f"[{section}/{'CARD' if is_card else 'INFERRED'}] {prefix}{text}"
            (card if is_card else inferred).append(line)
    out: list[str] = []
    used = 0
    for line in card + inferred:
        if used + len(line) + 1 > limit:
            if not out:
                out.append(line[:limit])
            break
        out.append(line)
        used += len(line) + 1
    return '\n'.join(out)


def _clip(value: Any, limit: int) -> str:
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + '…'


def _scene_line(scene: dict | None) -> str:
    if not isinstance(scene, dict):
        return ''
    parts = []
    for key in ('time', 'location'):
        value = scene.get(key)
        if value and value not in ('未明', '待确认'):
            parts.append(f'{key}: {value}')
    onstage = scene.get('onstage_npcs') or []
    if isinstance(onstage, list) and onstage:
        parts.append('onstage: ' + ', '.join(str(item) for item in onstage[:6]))
    return _clip('; '.join(parts), SCENE_CHAR_LIMIT)


def should_check(user_text: str) -> bool:
    text = str(user_text or '').strip()
    if len(text) < 2:
        return False
    return not any(pattern.search(text) for pattern in _SKIP_PATTERNS)


def build_user_prompt(bible_text: str, user_text: str, scene: dict | None = None) -> str:
    blocks = [f'WORLD BIBLE:\n{bible_text}']
    scene_line = _scene_line(scene)
    if scene_line:
        blocks.append(f'CURRENT SCENE: {scene_line}')
    blocks.append(f'PLAYER ACTION: {_clip(user_text, ACTION_CHAR_LIMIT)}')
    return '\n\n'.join(blocks)


_RULE_TAG = re.compile(r'^\s*\[[A-Za-z_]+/(?:CARD|INFERRED)\]\s*(?:\([^)]{1,30}\)\s*)?')


def normalize_result(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get('verdict', '') or '').strip().lower().replace('-', '_').replace(' ', '_')
    if verdict not in VERDICTS:
        return None
    cost = _clip(payload.get('cost', ''), FIELD_CHAR_LIMIT) if verdict == VERDICT_COST else ''
    return {
        'verdict': verdict,
        'rule': _clip(_RULE_TAG.sub('', str(payload.get('rule', '') or '')), FIELD_CHAR_LIMIT),
        'cost': cost,
        'narrator_hint': _clip(payload.get('narrator_hint', ''), FIELD_CHAR_LIMIT),
    }


def _bible_hash(bible_text: str) -> str:
    return hashlib.sha1(bible_text.encode('utf-8')).hexdigest()[:12]


def run_world_check(user_text: str, scene: dict | None = None, *, bible: dict | None = None) -> dict:
    """Classify one player action. Never raises.

    Returns ``{"status": ..., "ran": bool, ...}``. When ``ran`` is true the
    result also carries ``verdict``, ``rule``, ``cost``, ``narrator_hint``.
    """
    result: dict[str, Any] = {'ran': False}
    if not world_check_enabled():
        return {**result, 'status': 'skipped:disabled'}
    if not should_check(user_text):
        return {**result, 'status': 'skipped:not_an_action'}
    bible = bible if bible is not None else load_world_bible()
    bible_text = compact_bible(bible) if bible else ''
    if not bible_text:
        return {**result, 'status': 'skipped:no_world_bible'}
    runtime = get_role_runtime(WORLD_CHECK_ROLE)
    if runtime.get('provider') != 'llm':
        return {**result, 'status': 'skipped:role_not_llm'}
    try:
        model_cfg = dict(resolve_provider_model(runtime.get('model_role') or WORLD_CHECK_ROLE))
    except Exception as err:
        if 'NEBIUS_API_KEY' in str(err) and not nebius_api_key_configured():
            return {**result, 'status': 'skipped:no_nebius_api_key'}
        return {**result, 'status': f'error:config:{_clip(err, 120)}'}
    model_cfg['stream'] = False

    started = time.monotonic()
    try:
        reply, usage = call_model(model_cfg, WORLD_CHECK_SYSTEM, build_user_prompt(bible_text, user_text, scene))
    except Exception as err:
        logger.warning('world_check: model call failed: %s', err)
        return {
            **result,
            'status': f'error:call:{_clip(f"{type(err).__name__}: {err}", 160)}',
            'latency_ms': int((time.monotonic() - started) * 1000),
        }
    latency_ms = int((time.monotonic() - started) * 1000)
    usage = usage if isinstance(usage, dict) else {}
    base = {
        'latency_ms': latency_ms,
        'model': str(usage.get('model') or (model_cfg.get('model') or {}).get('id', '') or ''),
        'usage': {key: usage[key] for key in ('input_tokens', 'output_tokens', 'finish_reason') if key in usage},
        'bible_hash': _bible_hash(bible_text),
    }
    if usage.get('finish_reason') == 'length':
        return {**result, **base, 'status': 'error:truncated'}
    try:
        payload = parse_json_response(reply)
    except Exception:
        payload = None
    normalized = normalize_result(payload)
    if not normalized:
        return {**result, **base, 'status': 'error:unparsed', 'raw': _clip(reply, 200)}
    return {**base, **normalized, 'ran': True, 'status': 'ok'}


_VERDICT_LABEL_ZH = {
    VERDICT_ALLOWED: '允许',
    VERDICT_COST: '允许，但有代价',
    VERDICT_OUT: '超出本世界',
}


def narrator_block(result: dict | None) -> str:
    """Hidden narrator guidance. Empty for ``allowed`` with no hint, or no result."""
    if not isinstance(result, dict) or not result.get('ran'):
        return ''
    verdict = result.get('verdict')
    hint = result.get('narrator_hint', '')
    if verdict == VERDICT_ALLOWED and not hint:
        return ''
    lines = ['【世界规则检查】', f"- 判定：{_VERDICT_LABEL_ZH.get(verdict, verdict)}"]
    if result.get('rule'):
        lines.append(f"- 依据：{result['rule']}")
    if verdict == VERDICT_COST and result.get('cost'):
        lines.append(f"- 代价：{result['cost']}（必须在正文中以世界内的方式兑现，不得轻描淡写或略过）")
    if verdict == VERDICT_OUT:
        lines.append('- 该行动所需之物在本世界不存在：不得把它写成现实发生；只保留可兼容的意图，收束为本世界内能发生的事。')
    if hint:
        lines.append(f'- 处理建议：{hint}')
    lines.append('- 本块为隐形指引，不得在正文中提及“世界规则检查”、判定或依据字样。')
    return '\n'.join(lines)


def public_summary(result: dict | None) -> dict:
    """Compact shape for the API response / frontend status panel."""
    if not isinstance(result, dict):
        return {'status': 'skipped:none', 'ran': False}
    keys = ('status', 'ran', 'verdict', 'rule', 'cost', 'narrator_hint', 'latency_ms', 'model')
    return {key: result[key] for key in keys if key in result}
