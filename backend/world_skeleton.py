"""World skeleton: the fixed slice of the world bible the narrator sees every turn.

World Check only tells the narrator about the one rule an action touches. The
skeleton is the standing baseline (where things are, who holds power, what
things cost, how people live) so long sessions do not drift away from the
world the user locked. It is deterministic, cheap (no model call), and size
capped so it never crowds the rest of the narrator prompt.

* Global part (``SKELETON_CHAR_LIMIT``): round-robin over the sections in
  ``SKELETON_SECTION_ORDER``, canon entries first, world-wide entries only.
* Regional part (``REGION_CHAR_LIMIT``): entries whose ``region`` matches the
  current scene location, so the narrator gets local texture where the scene is.
"""
from __future__ import annotations

import re
from typing import Any

try:
    from .world_check import is_canon_entry, load_world_bible
except ImportError:  # flat-script imports (backend/ on sys.path)
    from world_check import is_canon_entry, load_world_bible

SKELETON_CHAR_LIMIT = 600
REGION_CHAR_LIMIT = 400
ENTRY_CHAR_LIMIT = 160

SKELETON_SECTION_ORDER: tuple[str, ...] = (
    'era_feel',
    'geography',
    'factions',
    'economy_resources',
    'customs',
    'tech_supernatural_limits',
    'social_system',
    'taboos',
)
SECTION_LABELS_ZH = {
    'era_feel': '时代',
    'geography': '地理',
    'factions': '势力',
    'economy_resources': '经济',
    'customs': '风土',
    'tech_supernatural_limits': '边界',
    'social_system': '社会',
    'taboos': '禁忌',
}
_UNKNOWN_LOCATIONS = {'', '未明', '待确认', 'unknown'}


def _clean(value: Any) -> str:
    return ' '.join(str(value or '').split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + '…'


def _ordered_entries(bible: dict, section: str) -> list[dict]:
    sections = bible.get('sections') if isinstance(bible.get('sections'), dict) else {}
    entries = [item for item in (sections.get(section) or []) if isinstance(item, dict) and _clean(item.get('text'))]
    locked = bible.get('locked') is True
    # Stable sort: canon first, original order otherwise.
    return sorted(entries, key=lambda item: 0 if is_canon_entry(item, locked=locked) else 1)


def _line(section: str, entry: dict) -> str:
    text = _clean(entry.get('text')).replace('{{user}}', '主角').replace('{{char}}', '角色')
    label = SECTION_LABELS_ZH.get(section, section)
    aliases = region_aliases(entry.get('region', ''))
    if aliases:
        label = f'{label}·{aliases[0]}'
    return f"- [{label}] {_clip(text, ENTRY_CHAR_LIMIT)}"


_ALIAS_SPLIT = re.compile(r'\s*[/／、,，|]\s*')


def region_aliases(region: str) -> list[str]:
    """"北方/燕云十六郡/幽州" -> ["北方", "燕云十六郡", "幽州"]."""
    return [alias for alias in _ALIAS_SPLIT.split(_clean(region).lower()) if alias]


def region_matches(region: str, location: str) -> bool:
    location = _clean(location).lower()
    if location in _UNKNOWN_LOCATIONS:
        return False
    for alias in region_aliases(region):
        if alias in location or (len(location) >= 2 and location in alias):
            return True
        # "岭南烟雨之地" vs location "岭南·七十二峰": match on a shared 2-char CJK head.
        cjk = re.sub(r'[^\u3400-\u9fff]', '', alias)
        if len(cjk) >= 3 and cjk[:2] in location:
            return True
    return False


def _take(candidates: list[tuple[str, dict]], limit: int, used_ids: set[int]) -> list[str]:
    lines: list[str] = []
    total = 0
    for section, entry in candidates:
        if id(entry) in used_ids:
            continue
        line = _line(section, entry)
        if total + len(line) + 1 > limit:
            continue
        lines.append(line)
        used_ids.add(id(entry))
        total += len(line) + 1
    return lines


def build_world_skeleton(bible: dict | None, location: str = '') -> str:
    """Return the narrator block, or '' when there is no bible."""
    if not isinstance(bible, dict) or not isinstance(bible.get('sections'), dict):
        return ''
    per_section = {name: _ordered_entries(bible, name) for name in SKELETON_SECTION_ORDER}
    used: set[int] = set()

    # Global: one entry per section per round, until the cap. World-wide
    # entries go first; regional ones only fill what budget is left.
    def _rounds(pick) -> list[tuple[str, dict]]:
        pools = {name: [item for item in per_section[name] if pick(item)] for name in SKELETON_SECTION_ORDER}
        depth = max((len(items) for items in pools.values()), default=0)
        return [
            (name, pools[name][index])
            for index in range(depth)
            for name in SKELETON_SECTION_ORDER
            if index < len(pools[name])
        ]

    global_first = _rounds(lambda item: not _clean(item.get('region')))
    global_lines = _take(global_first, SKELETON_CHAR_LIMIT, used)
    spare = SKELETON_CHAR_LIMIT - sum(len(line) + 1 for line in global_lines)
    regional_pool = [
        pair for pair in _rounds(lambda item: bool(_clean(item.get('region'))))
        if not region_matches(pair[1].get('region', ''), location)
    ]
    global_lines += _take(regional_pool, spare, used)

    regional = [
        (name, item)
        for name in SKELETON_SECTION_ORDER
        for item in per_section[name]
        if region_matches(item.get('region', ''), location)
    ]
    region_lines = _take(regional, REGION_CHAR_LIMIT, used)

    if not global_lines and not region_lines:
        return ''
    locked = bible.get('locked') is True
    head = (
        '【世界骨架】\n'
        + ('本块是用户已锁定的世界正史' if locked else '本块是当前世界设定的基础格局')
        + '：地理、势力、物价、风土与边界必须与之一致，不得改写、替换或另起一套；'
        '未提及的细节可以补充，但不能与之冲突。本块为隐形资料，不要在正文中复述或列举。'
    )
    parts = [head, *global_lines]
    if region_lines:
        place = _clip(_clean(location), 40)
        parts.append(f'当前区域（{place}）：')
        parts.extend(region_lines)
    return '\n'.join(parts)


def world_skeleton_for_scene(scene: dict | None = None, *, bible: dict | None = None) -> str:
    """Load the active bible and build the skeleton. Never raises."""
    try:
        bible = bible if bible is not None else load_world_bible()
        location = scene.get('location', '') if isinstance(scene, dict) else ''
        return build_world_skeleton(bible, location)
    except Exception:
        return ''
