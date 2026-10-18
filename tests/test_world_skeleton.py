#!/usr/bin/env python3
"""World skeleton tests: deterministic, no model calls."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import narrator_input  # noqa: E402
import world_skeleton as ws  # noqa: E402


def _bible(locked=False):
    return {
        'locked': locked,
        'sections': {
            'era_feel': [{'text': '王朝末年，藩镇割据，江湖与官府并存。', 'source': 'inferred'}],
            'geography': [
                {'text': '血河自西向东，是南北分界。', 'source': 'inferred'},
                {'text': '岭南多雾多雨，山道湿滑，七十二峰间只有栈道相通。', 'source': 'inferred', 'region': '岭南'},
            ],
            'factions': [{'text': '北方霸派占据燕云十六郡。', 'source': 'card'}],
            'economy_resources': [
                {'text': '一碗阳春面三文钱，一匹驽马二十两。', 'source': 'inferred'},
                {'text': '岭南以茶与药材换北地铁器。', 'source': 'inferred', 'region': '岭南'},
            ],
            'customs': [{'text': '岭南茶楼清晨开市，客人以叩指代谢。', 'source': 'inferred', 'region': '岭南'}],
            'taboos': [{'text': '私铸钱者斩。', 'source': 'user'}],
        },
    }


def test_global_skeleton_puts_world_wide_first_and_labels_other_regions():
    text = ws.build_world_skeleton(_bible(), '幽州城外')
    assert text.startswith('【世界骨架】')
    assert '一碗阳春面三文钱' in text and '北方霸派' in text and '私铸钱者斩' in text
    assert '当前区域' not in text
    lines = [line for line in text.split('\n') if line.startswith('- ')]
    first_regional = next(i for i, line in enumerate(lines) if '·岭南]' in line)
    assert all('·' not in line.split(']')[0] for line in lines[:first_regional])
    assert '- [风土·岭南] 岭南茶楼清晨开市' in text


def test_region_detail_follows_scene_location():
    text = ws.build_world_skeleton(_bible(), '岭南·七十二峰')
    assert '当前区域（岭南·七十二峰）' in text
    assert '叩指代谢' in text and '栈道相通' in text


def test_skeleton_respects_caps_and_locked_wording():
    bible = _bible(locked=True)
    bible['sections']['geography'] += [{'text': '山' * 150, 'source': 'inferred'} for _ in range(20)]
    text = ws.build_world_skeleton(bible, '')
    body = [line for line in text.split('\n') if line.startswith('- ')]
    assert sum(len(line) + 1 for line in body) <= ws.SKELETON_CHAR_LIMIT
    assert '用户已锁定的世界正史' in text


def test_empty_or_missing_bible_gives_nothing():
    assert ws.build_world_skeleton({}, '') == ''
    assert ws.build_world_skeleton(None, '') == ''
    assert ws.world_skeleton_for_scene({'location': 'x'}, bible={'sections': {}}) == ''


def test_region_matching():
    assert ws.region_matches('岭南', '岭南·七十二峰')
    assert ws.region_matches('岭南烟雨之地', '岭南七十二峰')
    assert not ws.region_matches('岭南', '待确认')
    assert not ws.region_matches('', '岭南')
    assert ws.region_matches('北方/燕云十六郡/幽州', '幽州城外')
    assert not ws.region_matches('南方', '幽州城外')
    assert not ws.region_matches('北方/燕云十六郡/幽州', '岭南')


def test_narrator_input_carries_skeleton_after_setting_lock():
    skeleton = ws.build_world_skeleton(_bible(), '')
    system_prompt, _ = narrator_input.build_narrator_input({'world_skeleton': skeleton, 'scene_facts': {}}, '我走进茶楼')
    assert '【世界骨架】' in system_prompt
    assert system_prompt.index('【世界设定锁】') < system_prompt.index('【世界骨架】')
