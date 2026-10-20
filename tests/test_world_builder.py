#!/usr/bin/env python3
"""World Builder tests. No network: the Nemotron call is always mocked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import card_importer as ci  # noqa: E402
import lorebook_distiller  # noqa: E402
import world_builder as wb  # noqa: E402
from character_assets import clear_character_override_root, set_character_override_root  # noqa: E402


CORE = {
    'name': '南诏行',
    'coreDescription': {
        'title': '南诏行',
        'tagline': '南诏国边境，战乱将起。',
        'summary': '一名寻找仙缘的少年侠客，在南诏边境卷入国教纷争。',
    },
    'tags': ['武侠'],
}
LOREBOOK = {
    'entries': [
        {'id': '1', 'title': '南诏地理', 'content': '南诏是一个边境小国，与大理多有摩擦。苍山之下有洱海。', 'entryType': 'place'},
        {'id': '2', 'title': '拜月教', 'content': '拜月教是南诏国教，势力遍布朝野。', 'entryType': 'faction'},
        {'id': '3', 'title': '世界基础规则', 'content': '本世界不存在火器。修炼需要以灵石为代价。', 'entryType': 'rule'},
        {'id': '4', 'title': 'NPC：阿奴', 'content': '苗族少女，拜月教圣女。', 'entryType': 'npc'},
        {'id': '5', 'title': '叙述规则', 'content': '不得替玩家做决定。', 'entryType': 'rule'},
    ],
}


def _llm_reply(sections: dict, setting_type: str = 'wuxia') -> str:
    return json.dumps({'setting_type': setting_type, 'sections': sections}, ensure_ascii=False)


@pytest.fixture
def nebius_key(monkeypatch):
    monkeypatch.setenv('NEBIUS_API_KEY', 'test-key-not-real')
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)


def _all_entries(bible: dict) -> list[dict]:
    return [item for items in bible['sections'].values() for item in items]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_world_bible_schema_with_mocked_llm(monkeypatch, nebius_key):
    reply = _llm_reply({
        'economy_resources': [{'title': '灵石', 'text': '一块下品灵石约值十两银，边境集市以银两与盐巴交易。'}],
        'social_system': [{'title': '土司', 'text': '边境由世袭土司治理，国教祭司可否决土司的裁决。'}],
    })
    monkeypatch.setattr(wb, '_call_world_builder_llm', lambda system, user: (reply, {'model': 'nvidia/Nemotron-3-Ultra-550b-a55b'}))

    bible = wb.build_world_bible(CORE, LOREBOOK)

    assert bible['version'] == wb.WORLD_BIBLE_VERSION
    assert list(bible['sections'].keys()) == list(wb.SECTIONS)
    assert set(wb.SECTIONS) == {
        'geography', 'factions', 'social_system', 'tech_supernatural_limits',
        'economy_resources', 'customs', 'taboos', 'era_feel',
    }
    entries = _all_entries(bible)
    assert entries
    for entry in entries:
        assert isinstance(entry['text'], str) and entry['text']
        assert entry['source'] in {'card', 'inferred'}
    assert bible['status']['llm'] == 'ok'
    assert bible['setting_type'] == 'wuxia'
    assert bible['models']['inferred'] == 'nvidia/Nemotron-3-Ultra-550b-a55b'
    assert bible['counts']['inferred'] == 2
    # Card entries come from the card and point back at lorebook ids.
    card_entries = [item for item in entries if item['source'] == 'card']
    assert any(item['source_entry_ids'] == ['1'] for item in bible['sections']['geography'])
    assert any(item['source_entry_ids'] == ['2'] for item in bible['sections']['factions'])
    assert all(item['source_entry_ids'] for item in card_entries)
    # NPC and narrator-meta entries are not world facts.
    assert not any('阿奴' in item['text'] for item in entries)
    assert not any('玩家' in item['text'] for item in entries)
    # Card prohibitions are surfaced as taboos.
    assert any('不存在火器' in item['text'] for item in bible['sections']['taboos'])


def test_card_only_bible_when_no_nebius_key(monkeypatch):
    monkeypatch.delenv('NEBIUS_API_KEY', raising=False)
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)

    def _boom(system, user):
        raise AssertionError('LLM must not be called without NEBIUS_API_KEY')

    monkeypatch.setattr(wb, '_call_world_builder_llm', _boom)
    bible = wb.build_world_bible(CORE, LOREBOOK)
    assert bible['status']['llm'] == 'skipped:no_nebius_api_key'
    assert bible['status']['tavily'] == 'skipped:no_key'
    assert bible['counts']['inferred'] == 0
    assert bible['counts']['card'] > 0


# ---------------------------------------------------------------------------
# card wins
# ---------------------------------------------------------------------------

def test_merge_card_wins_over_inferred():
    card_sections = wb.extract_card_sections(CORE, LOREBOOK)
    card_snapshot = json.loads(json.dumps(card_sections, ensure_ascii=False))
    inferred = {
        # contradicts the card's "no firearms" rule
        'tech_supernatural_limits': [
            {'title': '军械', 'text': '边军设有火器营，配发火铳。'},
            {'title': '修炼代价', 'text': '每突破一个境界需消耗百枚灵石，且会折损寿元三年。'},
        ],
        # collides with a card title -> may not replace the card entry
        'factions': [{'title': '拜月教', 'text': '拜月教其实早已式微，只剩几名老祭司。'}],
        # near-verbatim restatement of a card entry
        'geography': [
            {'title': '边境', 'text': '南诏是一个边境小国，与大理多有摩擦。'},
            # model tries to claim card provenance
            {'title': '雨季', 'text': '每年五月至十月为雨季，山道常被泥石流阻断。', 'source': 'card'},
        ],
    }

    merged, dropped = wb.merge_world_bible(card_sections, inferred)

    # Card entries are untouched and come first in every section.
    for section in wb.SECTIONS:
        card_part = [item for item in merged[section] if item['source'] == 'card']
        assert card_part == card_snapshot[section]
        assert merged[section][:len(card_part)] == card_part

    reasons = {item['title']: item['reason'] for item in dropped}
    assert reasons['军械'] == 'contradicts_card:火器'
    assert reasons['拜月教'] == 'title_collides_with_card'
    assert reasons['边境'] == 'duplicates_card'

    kept = {item['title']: item for items in merged.values() for item in items if item['source'] == 'inferred'}
    assert set(kept) == {'修炼代价', '雨季'}
    # Provenance is assigned by the merge, never trusted from the model.
    assert kept['雨季']['source'] == 'inferred'


def test_merge_allows_inferred_entry_that_restates_a_card_limit():
    card_sections = wb.extract_card_sections(CORE, LOREBOOK)
    merged, dropped = wb.merge_world_bible(card_sections, {
        'taboos': [{'title': '私造兵器', 'text': '民间严禁私藏火器图纸，违者流放。'}],
    })
    assert not dropped
    assert any(item['title'] == '私造兵器' and item['source'] == 'inferred' for item in merged['taboos'])


def test_merge_english_negation_and_section_limit():
    card_sections = wb.empty_sections()
    card_sections['tech_supernatural_limits'].append({'title': 'Rules', 'text': 'Magic does not exist in this world.', 'source': 'card'})
    inferred = {
        'tech_supernatural_limits': [{'title': 'Court mages', 'text': 'Court mages heal nobles with magic.'}],
        'factions': [{'title': f'Item {i}', 'text': text} for i, text in enumerate([
            'Salt is taxed at the river gates.',
            'Iron tools are rented by the season.',
            'Grain prices double every winter.',
            'Silver coins are clipped by smugglers.',
            'Horses belong only to garrison officers.',
            'Spice caravans pay tolls in pepper.',
        ])],
    }
    merged, dropped = wb.merge_world_bible(card_sections, inferred)
    assert any(item['reason'] == 'contradicts_card:magic' for item in dropped)
    assert len(merged['factions']) == wb.inferred_limit('factions') == 4
    assert sum(1 for item in dropped if item['reason'] == 'section_limit') == 2


def test_merge_ignores_malformed_llm_sections():
    card_sections = wb.extract_card_sections(CORE, LOREBOOK)
    merged, dropped = wb.merge_world_bible(card_sections, {
        'geography': 'not a list',
        'unknown_section': [{'text': 'ignored'}],
        'era_feel': [None, 42, {'title': 'no text'}, '一个纯字符串条目也能接受。'],
    })
    assert not dropped
    assert [item['text'] for item in merged['era_feel'] if item['source'] == 'inferred'] == ['一个纯字符串条目也能接受。']
    assert 'unknown_section' not in merged


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_llm_error_degrades_to_card_only(monkeypatch, nebius_key):
    def _fail(system, user):
        raise RuntimeError('token factory unavailable')

    monkeypatch.setattr(wb, '_call_world_builder_llm', _fail)
    bible = wb.build_world_bible(CORE, LOREBOOK)
    assert bible['status']['llm'].startswith('error:RuntimeError')
    assert bible['counts']['inferred'] == 0
    assert bible['counts']['card'] > 0


def test_llm_garbage_json_degrades_to_card_only(monkeypatch, nebius_key):
    monkeypatch.setattr(wb, '_call_world_builder_llm', lambda system, user: ('I cannot comply', {}))
    bible = wb.build_world_bible(CORE, LOREBOOK)
    assert bible['status']['llm'].startswith('error:')
    assert bible['counts']['card'] > 0


def _raw_card() -> dict:
    return {
        'spec': 'chara_card_v2',
        'spec_version': '2.0',
        'data': {
            'name': '南诏行',
            'description': '一名寻找仙缘的少年侠客。',
            'personality': '活泼机敏。',
            'scenario': '南诏国边境，战乱将起。',
            'first_mes': '你在客栈醒来。',
            'character_book': {
                'entries': [
                    {'id': 1, 'keys': ['南诏'], 'comment': '南诏地理', 'content': '南诏是一个边境小国，与大理多有摩擦。', 'enabled': True, 'constant': True},
                    {'id': 2, 'keys': ['拜月教'], 'comment': '拜月教势力', 'content': '拜月教是南诏国教，势力遍布朝野。', 'enabled': True},
                    {'id': 3, 'keys': ['火器'], 'comment': '世界规则', 'content': '本世界不存在火器。', 'enabled': True, 'constant': True},
                ],
            },
        },
    }


@pytest.fixture
def import_root(tmp_path, monkeypatch):
    # Keep the lorebook distiller off the network too.
    def _no_distill_llm(system, user):
        raise RuntimeError('no network in tests')

    monkeypatch.setattr(lorebook_distiller, '_call_distill_llm', _no_distill_llm)
    target = tmp_path / 'character'
    target.mkdir()
    set_character_override_root(target)
    try:
        yield target
    finally:
        clear_character_override_root()


def test_card_import_survives_world_builder_llm_error(monkeypatch, nebius_key, import_root):
    def _fail(system, user):
        raise TimeoutError('nemotron timed out')

    monkeypatch.setattr(wb, '_call_world_builder_llm', _fail)
    result = ci.import_card_bundle(_raw_card())

    assert result['success'] is True
    assert result['world_bible']['success'] is True
    assert result['world_bible']['status']['llm'].startswith('error:TimeoutError')
    bible_path = import_root / wb.WORLD_BIBLE_FILE
    assert bible_path.exists()
    assert (import_root / lorebook_distiller.FOUNDATION_FILE).exists()
    bible = json.loads(bible_path.read_text(encoding='utf-8'))
    assert bible['counts']['card'] > 0
    assert bible['character_name'] == '南诏行'


def test_card_import_survives_unexpected_world_builder_crash(monkeypatch, import_root):
    def _crash(*args, **kwargs):
        raise ValueError('bug in world builder')

    monkeypatch.setattr(wb, 'build_world_bible', _crash)
    result = ci.import_card_bundle(_raw_card())
    assert result['success'] is True
    assert result['world_bible']['success'] is False
    assert 'bug in world builder' in result['world_bible']['error']
    assert (import_root / 'character-data.json').exists()


def test_card_import_writes_inferred_entries(monkeypatch, nebius_key, import_root):
    reply = _llm_reply({'economy_resources': [{'title': '盐', 'text': '井盐由土司专营，私盐贩子按律杖八十。'}]})
    monkeypatch.setattr(wb, '_call_world_builder_llm', lambda system, user: (reply, {'model': 'mock'}))
    result = ci.import_card_bundle(_raw_card())
    assert result['world_bible']['inferred_entries'] == 1
    bible = json.loads((import_root / wb.WORLD_BIBLE_FILE).read_text(encoding='utf-8'))
    assert bible['sections']['economy_resources'][-1]['source'] == 'inferred'


# ---------------------------------------------------------------------------
# rebuild: cache, lock, force
# ---------------------------------------------------------------------------

def _write_source(root: Path) -> None:
    (root / 'character-data.json').write_text(json.dumps(CORE, ensure_ascii=False), encoding='utf-8')
    (root / 'lorebook.json').write_text(json.dumps(LOREBOOK, ensure_ascii=False), encoding='utf-8')


def test_rebuild_cache_lock_and_force(monkeypatch, nebius_key, tmp_path):
    _write_source(tmp_path)
    calls = []

    def _llm(system, user):
        calls.append(user)
        return _llm_reply({'era_feel': [{'title': '乱世', 'text': '各国边军频繁换防，商旅结队而行。'}]}), {'model': 'mock'}

    monkeypatch.setattr(wb, '_call_world_builder_llm', _llm)

    assert wb.rebuild_world_bible(tmp_path)['action'] == 'rebuilt'
    assert wb.rebuild_world_bible(tmp_path)['action'] == 'kept_cached'
    assert len(calls) == 1

    path = tmp_path / wb.WORLD_BIBLE_FILE
    bible = json.loads(path.read_text(encoding='utf-8'))
    bible['locked'] = True
    bible['status']['llm'] = 'error:previous'
    path.write_text(json.dumps(bible, ensure_ascii=False), encoding='utf-8')
    assert wb.rebuild_world_bible(tmp_path)['action'] == 'kept_locked'
    assert len(calls) == 1

    assert wb.rebuild_world_bible(tmp_path, force=True)['action'] == 'rebuilt'
    assert len(calls) == 2

    # A different card invalidates both cache and lock.
    bible = json.loads(path.read_text(encoding='utf-8'))
    bible['locked'] = True
    path.write_text(json.dumps(bible, ensure_ascii=False), encoding='utf-8')
    changed = dict(LOREBOOK)
    changed['entries'] = LOREBOOK['entries'] + [{'id': '9', 'title': '新地点', 'content': '新增的雪山关隘。', 'entryType': 'place'}]
    (tmp_path / 'lorebook.json').write_text(json.dumps(changed, ensure_ascii=False), encoding='utf-8')
    assert wb.rebuild_world_bible(tmp_path)['action'] == 'rebuilt'
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Tavily hook
# ---------------------------------------------------------------------------

def test_tavily_skips_without_key_or_for_fiction(monkeypatch):
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    notes, status = wb.fetch_tavily_research(CORE, wb.empty_sections())
    assert (notes, status) == ([], 'skipped:no_key')

    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    fantasy = {'name': 'x', 'coreDescription': {'tagline': '修仙世界的历史长河'}, 'tags': []}
    notes, status = wb.fetch_tavily_research(fantasy, wb.empty_sections())
    assert (notes, status) == ([], 'skipped:fictional_setting')


def test_tavily_called_for_historical_setting(monkeypatch):
    import safe_http

    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    captured = {}

    def _fake_request(url, *, method='GET', headers=None, body=None, timeout=20.0):
        captured.update(url=url, method=method, headers=headers, body=json.loads(body or b'{}'))
        payload = {'results': [{'title': '民国上海的米价', 'url': 'https://example.org/rice', 'content': '1930年代上海一石米约值十元。'}]}
        return 200, {}, json.dumps(payload, ensure_ascii=False).encode('utf-8')

    monkeypatch.setattr(safe_http, 'safe_request', _fake_request)
    historical = {'name': 'x', 'coreDescription': {'tagline': '民国二十年的上海滩'}, 'tags': []}
    notes, status = wb.fetch_tavily_research(historical, wb.empty_sections())
    assert status == 'ok'
    assert notes[0]['url'] == 'https://example.org/rice'
    assert captured['url'] == wb.TAVILY_SEARCH_URL
    assert captured['method'] == 'POST'
    assert captured['headers']['Authorization'] == 'Bearer tvly-test'
    assert '民国二十年的上海滩' in captured['body']['query']


# ---------------------------------------------------------------------------
# English cards (regressions found in the 2026-10-18 live run)
# ---------------------------------------------------------------------------

EN_CARD_SECTIONS = {
    'geography': [{
        'id': 'card-1', 'title': 'Geography: the Greywater ford', 'source': 'card', 'source_entry_ids': ['1'],
        'text': "The Greywater River is crossable only at Emberwell's ford, and only when the water is low. "
                'A rope ferry runs when the ford floods, at two silver bits per head.',
    }],
    'tech_supernatural_limits': [{
        'id': 'card-2', 'title': 'World rule: technology', 'source': 'card', 'source_entry_ids': ['5'],
        'text': 'There is no gunpowder in this world. Firearms do not exist.',
    }],
}


def test_english_inferred_entries_are_not_flagged_as_duplicates():
    inferred = {
        'geography': [
            # New detail that shares common English letters/bigrams but no content.
            {'title': 'Town layout', 'text': 'Emberwell clusters on the south bank around the inn; mud streets radiate from the stone ramp.'},
            # A genuine restatement of the card entry.
            {'title': 'Ford', 'text': "The Greywater River is crossable only at Emberwell's ford when the water is low; the rope ferry costs two silver bits per head."},
        ],
        'tech_supernatural_limits': [
            {'title': 'Muskets', 'text': 'Muskets are carried by the town watch.'},
        ],
    }
    merged, dropped = wb.merge_world_bible(EN_CARD_SECTIONS, inferred)
    kept = [item['title'] for item in merged['geography'] if item['source'] == 'inferred']
    assert kept == ['Town layout']
    reasons = {item['title']: item['reason'] for item in dropped}
    assert reasons['Ford'] == 'duplicates_card'
    assert 'Muskets' not in reasons or reasons['Muskets'] != 'duplicates_card'


def test_english_keywords_match_whole_words():
    # 'caster' must not count as the social keyword 'caste'.
    section = wb.classify_section(
        'World rule: hedge-craft',
        'Magic is rare and costly. Small charms of warding cost the caster blood and days of fever.',
    )
    assert section == 'tech_supernatural_limits'
    assert wb.classify_section('Faction: the Ferrymen', 'The ferrymen are a guild that controls the landings.') == 'factions'


def test_truncated_llm_output_degrades_to_card_only(monkeypatch, nebius_key):
    reply = _llm_reply({'geography': [{'title': 'x', 'text': '苍山脚下有三座驿站。'}]})
    monkeypatch.setattr(wb, '_call_world_builder_llm', lambda system, user: (reply, {'finish_reason': 'length', 'output_tokens': 8000}))
    bible = wb.build_world_bible(CORE, LOREBOOK, use_tavily=False)
    assert bible['status']['llm'].startswith('error:ValueError: LLM output truncated')
    assert bible['counts']['inferred'] == 0


def test_world_builder_retries_without_response_format(monkeypatch, nebius_key):
    import io
    from urllib.error import HTTPError

    import llm_manager
    import model_config

    monkeypatch.setattr(llm_manager, 'get_role_runtime', lambda role: {'provider': 'llm', 'model_role': role})
    monkeypatch.setattr(wb, 'get_role_runtime', lambda role: {'provider': 'llm', 'model_role': role})
    monkeypatch.setattr(wb, 'resolve_provider_model', lambda role: {
        'provider': {'baseUrl': 'https://api.tokenfactory.nebius.com/v1', 'apiKey': 'k'},
        'model': {'id': model_config.NEMOTRON_ULTRA_MODEL},
        'temperature': 0.4, 'max_output_tokens': 8000, 'stream': False,
        'response_format': {'type': 'json_object'},
    })
    calls = []

    def _fake_call_model(config, system, user):
        calls.append(dict(config))
        if config.get('response_format'):
            raise HTTPError('u', 400, 'Bad Request', None, io.BytesIO(b'{"error": "response_format json_object is not supported"}'))
        return '{"sections": {}}', {'model': config['model']['id']}

    monkeypatch.setattr(wb, 'call_model', _fake_call_model)
    reply, usage = wb._call_world_builder_llm('sys', 'user')
    assert reply == '{"sections": {}}'
    assert len(calls) == 2 and 'response_format' not in calls[1]
    assert usage['response_format_fallback'] is True
    assert 'duration_ms' in usage


# ── UI edits (save_world_bible_edits) ───────────────────────────────────────

def _seed_bible(root):
    bible = {
        'version': 1, 'locked': False,
        'sections': {**wb.empty_sections(),
                     'taboos': [{'id': 'card-1', 'title': 'River', 'text': 'Never pull the drowned out after dark.', 'source': 'card'}],
                     'geography': [{'id': 'inf-1', 'title': '', 'text': 'Camps lie north.', 'source': 'inferred'}]},
        'counts': {'card': 1, 'inferred': 1},
    }
    (root / wb.WORLD_BIBLE_FILE).write_text(json.dumps(bible), encoding='utf-8')


def test_save_edits_keeps_sources_marks_edited_and_locks(tmp_path):
    _seed_bible(tmp_path)
    out = wb.save_world_bible_edits({
        'taboos': [{'id': 'card-1', 'text': 'Never pull the drowned out after sundown.'}, {'text': 'No fires on the ferry.'}],
        'geography': [],
    }, source_root=tmp_path)
    taboos = out['sections']['taboos']
    assert taboos[0]['source'] == 'card' and taboos[0]['edited'] is True and taboos[0]['title'] == 'River'
    assert taboos[1]['source'] == 'user' and taboos[1]['id'].startswith('user-')
    assert out['sections']['geography'] == []
    assert out['counts'] == {'card': 1, 'inferred': 0, 'user': 1}
    assert out['locked'] is True
    assert json.loads((tmp_path / wb.WORLD_BIBLE_FILE).read_text(encoding='utf-8'))['locked'] is True


def test_save_edits_unchanged_text_not_marked_and_omitted_sections_kept(tmp_path):
    _seed_bible(tmp_path)
    out = wb.save_world_bible_edits({'taboos': [{'id': 'card-1', 'text': 'Never pull the drowned out after dark.'}]}, source_root=tmp_path)
    assert 'edited' not in out['sections']['taboos'][0]
    assert out['sections']['geography'][0]['id'] == 'inf-1'


def test_lock_toggle_only(tmp_path):
    _seed_bible(tmp_path)
    assert wb.save_world_bible_edits(locked=True, source_root=tmp_path)['locked'] is True
    out = wb.save_world_bible_edits(locked=False, source_root=tmp_path)
    assert out['locked'] is False and out['sections']['taboos'][0]['id'] == 'card-1'


@pytest.mark.parametrize('sections,msg', [
    ({'magic': []}, 'unknown sections'),
    ({'taboos': 'x'}, 'must be a list'),
    ({'taboos': [{'text': 'x' * 601}]}, 'longer than'),
    ({'taboos': [{'text': f'r{i}'} for i in range(41)]}, 'more than'),
])
def test_save_edits_rejects_bad_input(tmp_path, sections, msg):
    _seed_bible(tmp_path)
    with pytest.raises(ValueError, match=msg):
        wb.save_world_bible_edits(sections, source_root=tmp_path)


def test_save_edits_without_bible_raises(tmp_path):
    with pytest.raises(ValueError, match='no world bible'):
        wb.save_world_bible_edits({'taboos': []}, source_root=tmp_path)


def test_locked_user_edits_survive_rebuild(tmp_path, monkeypatch):
    (tmp_path / 'character-data.json').write_text(json.dumps(CORE, ensure_ascii=False), encoding='utf-8')
    (tmp_path / 'lorebook.json').write_text(json.dumps(LOREBOOK, ensure_ascii=False), encoding='utf-8')
    monkeypatch.delenv('NEBIUS_API_KEY', raising=False)
    wb.rebuild_world_bible(tmp_path, use_tavily=False)
    wb.save_world_bible_edits({'era_feel': [{'text': 'Rain every evening.'}]}, source_root=tmp_path)
    report = wb.rebuild_world_bible(tmp_path, use_tavily=False)
    assert report['action'] == 'kept_locked'
    assert wb.load_world_bible(tmp_path)['sections']['era_feel'][0]['text'] == 'Rain every evening.'


def test_texture_sections_get_more_room_and_keep_region():
    inferred = {
        'customs': [
            {'title': f'C{i}', 'text': text, 'region': '岭南' if i % 2 else ''}
            for i, text in enumerate([
                '北地人早饭吃羊汤泡馍，冬日家家腌酸菜。',
                '岭南茶楼清晨开市，客人以叩指代谢。',
                '京畿官话为尊，市井以儿化音辨身份。',
                '漕帮船工端午赛龙舟，输家请全船喝酒。',
                '七十二峰门人见面行抱剑礼，不握手。',
                '边镇妇人头裹青布，出门必带短刀。',
                '江上渡口挂红灯笼表示夜里仍可渡河。',
                '山中猎户用骨哨互通讯息，三短一长为求援。',
            ])
        ],
    }
    merged, dropped = wb.merge_world_bible(wb.empty_sections(), inferred)
    assert len(merged['customs']) == 6
    assert sum(1 for item in dropped if item['reason'] == 'section_limit') == 2
    assert merged['customs'][1]['region'] == '岭南'
    assert 'region' not in merged['customs'][0]


def test_card_customs_are_classified():
    assert wb.classify_section('岭南风俗', '茶楼清晨开市，节庆时家家挂灯。') == 'customs'


def test_prompt_separates_physical_geography_and_asks_for_prices():
    system = wb.WORLD_BUILDER_SYSTEM
    assert '"customs": []' in system
    assert 'PHYSICAL world only' in system
    assert 'CONCRETE everyday prices' in system
    assert '"region"' in system and '北方/燕云十六郡/幽州' in system


def test_user_edit_can_set_and_clear_region(tmp_path):
    bible = wb.build_world_bible(CORE, LOREBOOK, use_llm=False, use_tavily=False)
    (tmp_path / wb.WORLD_BIBLE_FILE).write_text(json.dumps(bible, ensure_ascii=False), encoding='utf-8')
    saved = wb.save_world_bible_edits(
        {'customs': [{'text': '岭南茶楼清晨开市。', 'region': '岭南'}]}, source_root=tmp_path,
    )
    entry = saved['sections']['customs'][0]
    assert entry['region'] == '岭南' and entry['source'] == 'user'
    saved = wb.save_world_bible_edits(
        {'customs': [{'id': entry['id'], 'text': entry['text'], 'region': ''}]}, source_root=tmp_path,
    )
    assert 'region' not in saved['sections']['customs'][0]


# ---------------------------------------------------------------------------
# Tavily research planner (Nemotron picks queries, notes are cited by id)
# ---------------------------------------------------------------------------

WUXIA_CORE = {
    'name': '血河渡',
    'coreDescription': {'tagline': '武侠江湖', 'summary': '北方燕云十六郡与岭南七十二峰之间的江湖恩怨。'},
    'tags': ['武侠'],
}


def _planner_env(monkeypatch, reply):
    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    monkeypatch.setattr(wb, 'nebius_api_key_configured', lambda: True)
    monkeypatch.setattr(wb, 'get_role_runtime', lambda role: {'provider': 'llm', 'model_role': role})
    monkeypatch.setattr(wb, 'resolve_provider_model', lambda role: {'model': f'mock-{role}'})
    calls = []

    def _fake_call_model(config, system, user):
        calls.append((config.get('model'), system))
        return json.dumps(reply, ensure_ascii=False), {'model': config.get('model')}

    monkeypatch.setattr(wb, 'call_model', _fake_call_model)
    return calls


def test_planner_returns_capped_deduped_queries(monkeypatch):
    calls = _planner_env(monkeypatch, {
        'mode': 'analog',
        'analog': '北宋',
        'queries': [
            {'q': '北宋 客栈 住宿 价格', 'section': 'economy_resources', 'why': '物价'},
            {'q': '北宋 客栈 住宿 价格', 'section': 'economy_resources'},
            '宋代 驿站 马匹 日行',
            {'q': '宋代 市井 饮食 服饰', 'section': 'bogus'},
            {'q': '多余的第四条'},
        ],
    })
    queries, plan = wb.plan_research_queries(WUXIA_CORE, wb.empty_sections())
    assert [q['query'] for q in queries] == ['北宋 客栈 住宿 价格', '宋代 驿站 马匹 日行', '宋代 市井 饮食 服饰']
    assert queries[0]['section'] == 'economy_resources'
    assert queries[2]['section'] == ''
    assert plan['mode'] == 'analog' and plan['analog'] == '北宋' and plan['status'] == 'ok'
    assert calls[0][0] == 'mock-world_research'


def test_planner_none_mode_yields_no_queries(monkeypatch):
    _planner_env(monkeypatch, {'mode': 'none', 'queries': [{'q': 'should be ignored'}]})
    queries, plan = wb.plan_research_queries(WUXIA_CORE, wb.empty_sections())
    assert queries == [] and plan['mode'] == 'none'


def test_planner_failure_is_soft(monkeypatch):
    monkeypatch.setattr(wb, 'nebius_api_key_configured', lambda: True)
    monkeypatch.setattr(wb, 'get_role_runtime', lambda role: {'provider': 'llm', 'model_role': role})
    monkeypatch.setattr(wb, 'resolve_provider_model', lambda role: {'model': 'x'})
    monkeypatch.setattr(wb, 'call_model', lambda *a: ('not json at all', {}))
    queries, plan = wb.plan_research_queries(WUXIA_CORE, wb.empty_sections())
    assert queries == [] and plan['status'].startswith('error:')


def test_fetch_runs_each_planned_query_and_numbers_notes(monkeypatch):
    import safe_http

    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    seen = []

    def _fake_request(url, *, method='GET', headers=None, body=None, timeout=20.0):
        query = json.loads(body)['query']
        seen.append((query, json.loads(body)['max_results']))
        results = [
            {'title': f'{query} A', 'url': f'https://ex.org/{len(seen)}a', 'content': '一碗面三文钱。'},
            {'title': 'dup', 'url': 'https://ex.org/1a', 'content': 'duplicate url'},
        ]
        return 200, {}, json.dumps({'results': results}, ensure_ascii=False).encode('utf-8')

    monkeypatch.setattr(safe_http, 'safe_request', _fake_request)
    queries = [{'query': 'q1', 'section': 'economy_resources'}, {'query': 'q2', 'section': 'customs'}]
    notes, status = wb.fetch_tavily_research(WUXIA_CORE, wb.empty_sections(), queries)
    assert status == 'ok'
    assert seen == [('q1', wb.TAVILY_RESULTS_PER_QUERY), ('q2', wb.TAVILY_RESULTS_PER_QUERY)]
    assert [n['id'] for n in notes] == ['R1', 'R2']
    assert notes[1]['section'] == 'customs' and notes[1]['query'] == 'q2'


def test_build_cites_refs_and_drops_unknown_ones(monkeypatch, nebius_key):
    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    monkeypatch.setattr(wb, 'plan_research_queries', lambda core, sections: (
        [{'query': '北宋 物价', 'section': 'economy_resources', 'why': ''}],
        {'mode': 'analog', 'analog': '北宋', 'status': 'ok', 'model': 'mock-super'},
    ))
    notes = [{'id': 'R1', 'title': '宋代物价', 'url': 'https://ex.org/song', 'snippet': '一斗米约百文。', 'query': '北宋 物价', 'section': 'economy_resources'}]
    monkeypatch.setattr(wb, 'fetch_tavily_research', lambda core, sections, queries=None, era_terms=None: (notes, 'ok'))
    prompts = {}

    def _llm(system, user):
        prompts['user'] = user
        prompts['system'] = system
        return json.dumps({'setting_type': 'wuxia', 'sections': {
            'economy_resources': [
                {'title': '米价', 'text': '幽州一斗米约九十文，荒年翻倍。', 'refs': ['R1', 'R9']},
                {'title': '面价', 'text': '路边一碗汤面五文。', 'refs': 'r1'},
                {'title': '刀价', 'text': '寻常腰刀一贯二百文。'},
            ],
        }}, ensure_ascii=False), {'model': 'mock-ultra'}

    monkeypatch.setattr(wb, '_call_world_builder_llm', _llm)
    bible = wb.build_world_bible(WUXIA_CORE, {'entries': []})
    assert '[R1] (for economy_resources) 宋代物价' in prompts['user']
    assert 'mode=analog, reference era: 北宋' in prompts['user']
    assert 'refs' in prompts['system']
    econ = {item['title']: item for item in bible['sections']['economy_resources']}
    assert econ['米价']['refs'] == ['R1']
    assert econ['面价']['refs'] == ['R1']
    assert 'refs' not in econ['刀价']
    assert bible['research'] == [{'id': 'R1', 'title': '宋代物价', 'url': 'https://ex.org/song', 'query': '北宋 物价', 'cited': True}]
    assert bible['research_plan'] == {'mode': 'analog', 'analog': '北宋', 'model': 'mock-super', 'queries': ['北宋 物价']}
    assert bible['status']['tavily'] == 'ok'


def test_build_skips_tavily_when_planner_says_none(monkeypatch, nebius_key):
    monkeypatch.setenv('TAVILY_API_KEY', 'tvly-test')
    monkeypatch.setattr(wb, 'plan_research_queries', lambda core, sections: ([], {'mode': 'none', 'status': 'ok:no_queries(none)'}))

    def _no_fetch(*a, **k):
        raise AssertionError('Tavily must not be called')

    monkeypatch.setattr(wb, 'fetch_tavily_research', _no_fetch)
    monkeypatch.setattr(wb, '_call_world_builder_llm', lambda s, u: (json.dumps({'sections': {}}), {}))
    bible = wb.build_world_bible(WUXIA_CORE, {'entries': []})
    assert bible['status']['tavily'] == 'skipped:planner_none'
    assert bible['research'] == []


def test_taboos_only_explicit_prohibitions_with_context():
    content = (
        '北境:是大胤绝不可失的北方门户.\n'
        '外在: 血雾(无法完全控制的力量)环绕.\n'
        '核心业务:专精于挖掘见不得光的秘密.\n'
        '[系统指令:地理锁定]\n'
        '绮音阁、金鳞池均为神都独有的地标.\n'
        '严禁将它们出现在江南、西域等其他地区.\n'
        '天策府:\n'
        '本质:正道组织.严禁帮众欺压良善.\n'
        '梦蝶(艺名)\n'
        '逻辑锁:外界仅知艺名"梦蝶".公共场合严禁提及真名.\n'
    )
    got = wb._constraint_sentences(content, limit=5)
    assert got == [
        '绮音阁、金鳞池均为神都独有的地标. 严禁将它们出现在江南、西域等其他地区.',
        '天策府：严禁帮众欺压良善.',
    ]
