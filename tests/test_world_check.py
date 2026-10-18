#!/usr/bin/env python3
"""World Check tests. No network: call_model / config are always mocked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import narrator_input  # noqa: E402
import world_check as wc  # noqa: E402


BIBLE = {
    'sections': {
        'tech_supernatural_limits': [
            {'text': 'Frontier town circa 1870s: no electricity, no telegraph east of the river.', 'source': 'card'},
            {'text': 'Folk charms exist but need blood and fail often.', 'source': 'inferred'},
        ],
        'taboos': [
            {'text': 'Touching the drowned after dark is forbidden; the ferrymen shun anyone who does.', 'source': 'card'},
        ],
        'geography': [],
    },
}


@pytest.fixture
def fake_model(monkeypatch):
    calls = []
    state = {'reply': json.dumps({'verdict': 'allowed', 'rule': '', 'cost': '', 'narrator_hint': ''}), 'usage': {'model': 'nano', 'finish_reason': 'stop'}}

    monkeypatch.setenv(wc.ENABLE_ENV, '1')
    monkeypatch.setattr(wc, 'get_role_runtime', lambda role: {'provider': 'llm', 'model_role': role})
    monkeypatch.setattr(wc, 'resolve_provider_model', lambda role: {'provider': {'name': 'nebius-token-factory'}, 'model': {'id': 'nano'}, 'stream': False})

    def call(cfg, system, user):
        calls.append({'cfg': cfg, 'system': system, 'user': user})
        if isinstance(state['reply'], Exception):
            raise state['reply']
        return state['reply'], dict(state['usage'])

    monkeypatch.setattr(wc, 'call_model', call)
    return state, calls


def test_compact_bible_puts_card_first_and_respects_limit():
    text = wc.compact_bible(BIBLE)
    lines = text.splitlines()
    assert lines[0].startswith('[tech_supernatural_limits/CARD]')
    assert lines[1].startswith('[taboos/CARD]')
    assert lines[-1].startswith('[tech_supernatural_limits/INFERRED]')
    short = wc.compact_bible(BIBLE, limit=120)
    assert 'INFERRED' not in short and len(short) <= 120


def test_system_prompt_routes_taboos_to_cost_not_out_of_world():
    prompt = wc.WORLD_CHECK_SYSTEM
    assert 'allowed_with_cost, never out_of_world' in prompt
    assert 'impossibility' in prompt


def test_skips_without_bible(fake_model):
    _, calls = fake_model
    result = wc.run_world_check('I walk to the ferry.', bible={})
    assert result == {'ran': False, 'status': 'skipped:no_world_bible'}
    assert not calls


def test_skips_when_disabled_or_not_an_action(fake_model, monkeypatch):
    _, calls = fake_model
    assert wc.run_world_check('/regen', bible=BIBLE)['status'] == 'skipped:not_an_action'
    assert wc.run_world_check('(OOC: pause)', bible=BIBLE)['status'] == 'skipped:not_an_action'
    monkeypatch.setenv(wc.ENABLE_ENV, '0')
    assert wc.run_world_check('I walk to the ferry.', bible=BIBLE)['status'] == 'skipped:disabled'
    assert not calls


def test_missing_nebius_key_is_a_skip(fake_model, monkeypatch):
    def boom(role):
        raise RuntimeError('NEBIUS_API_KEY is not set; Nebius Token Factory role world_check is unavailable')
    monkeypatch.setattr(wc, 'resolve_provider_model', boom)
    monkeypatch.delenv('NEBIUS_API_KEY', raising=False)
    assert wc.run_world_check('I walk to the ferry.', bible=BIBLE)['status'] == 'skipped:no_nebius_api_key'


def test_cost_verdict_is_parsed_and_prompt_carries_scene(fake_model):
    state, calls = fake_model
    state['reply'] = json.dumps({
        'verdict': 'allowed-with-cost',
        'rule': 'Touching the drowned after dark is forbidden',
        'cost': 'The ferrymen will refuse you passage.',
        'narrator_hint': 'Let a ferryman see it happen.',
    })
    result = wc.run_world_check('I drag the drowned man ashore after dark.', {'location': 'riverbank', 'time': 'night', 'onstage_npcs': ['Mara']}, bible=BIBLE)
    assert result['ran'] is True and result['status'] == 'ok'
    assert result['verdict'] == 'allowed_with_cost'
    assert result['cost'] == 'The ferrymen will refuse you passage.'
    user = calls[0]['user']
    assert 'WORLD BIBLE:' in user and 'location: riverbank' in user and 'onstage: Mara' in user
    assert user.rstrip().endswith('PLAYER ACTION: I drag the drowned man ashore after dark.')


def test_cost_dropped_for_non_cost_verdicts(fake_model):
    state, _ = fake_model
    state['reply'] = json.dumps({'verdict': 'out_of_world', 'rule': 'no electricity', 'cost': 'x', 'narrator_hint': 'It is a pocket mirror.'})
    result = wc.run_world_check('I check my smartphone.', bible=BIBLE)
    assert result['verdict'] == 'out_of_world' and result['cost'] == ''


@pytest.mark.parametrize('reply,usage,status', [
    ('not json at all', {'finish_reason': 'stop'}, 'error:unparsed'),
    (json.dumps({'verdict': 'maybe'}), {'finish_reason': 'stop'}, 'error:unparsed'),
    (json.dumps({'verdict': 'allowed'}), {'finish_reason': 'length'}, 'error:truncated'),
])
def test_bad_model_output_fails_open(fake_model, reply, usage, status):
    state, _ = fake_model
    state['reply'], state['usage'] = reply, usage
    result = wc.run_world_check('I walk to the ferry.', bible=BIBLE)
    assert result['ran'] is False and result['status'] == status
    assert wc.narrator_block(result) == ''


def test_call_exception_fails_open(fake_model):
    state, _ = fake_model
    state['reply'] = TimeoutError('read timed out')
    result = wc.run_world_check('I walk to the ferry.', bible=BIBLE)
    assert result['ran'] is False and result['status'].startswith('error:call:TimeoutError')


def test_narrator_block_shapes():
    assert wc.narrator_block({'ran': True, 'verdict': 'allowed', 'narrator_hint': ''}) == ''
    cost = wc.narrator_block({'ran': True, 'verdict': 'allowed_with_cost', 'rule': 'r', 'cost': 'shunned', 'narrator_hint': 'h'})
    assert cost.startswith('【世界规则检查】') and '代价：shunned' in cost and '处理建议：h' in cost
    out = wc.narrator_block({'ran': True, 'verdict': 'out_of_world', 'rule': 'no electricity', 'cost': '', 'narrator_hint': ''})
    assert '不得把它写成现实发生' in out


def test_public_summary_hides_usage_and_raw():
    summary = wc.public_summary({'ran': True, 'status': 'ok', 'verdict': 'allowed', 'usage': {'input_tokens': 9}, 'raw': 'x', 'bible_hash': 'abc'})
    assert summary == {'ran': True, 'status': 'ok', 'verdict': 'allowed'}


def test_load_world_bible_from_source_root(tmp_path):
    assert wc.load_world_bible(tmp_path) == {}
    (tmp_path / wc.WORLD_BIBLE_FILE).write_text(json.dumps(BIBLE), encoding='utf-8')
    assert wc.load_world_bible(tmp_path)['sections']['taboos'][0]['source'] == 'card'
    (tmp_path / wc.WORLD_BIBLE_FILE).write_text('{broken', encoding='utf-8')
    assert wc.load_world_bible(tmp_path) == {}


def test_narrator_input_injects_world_check_block(monkeypatch):
    block = wc.narrator_block({'ran': True, 'verdict': 'out_of_world', 'rule': 'no electricity', 'narrator_hint': 'h'})
    system_prompt, _ = narrator_input.build_narrator_input({}, 'I check my phone.', world_check_block=block)
    assert '【世界规则检查】' in system_prompt
    plain, _ = narrator_input.build_narrator_input({}, 'I check my phone.')
    assert '【世界规则检查】' not in plain


def test_rule_section_tag_is_stripped():
    out = wc.normalize_result({'verdict': 'allowed', 'rule': '[taboos/CARD] Firearms do not exist.'})
    assert out['rule'] == 'Firearms do not exist.'
