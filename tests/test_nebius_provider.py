#!/usr/bin/env python3
"""Nebius Token Factory provider wiring. No network: call_model is mocked."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import llm_manager  # noqa: E402
import model_config  # noqa: E402
import world_builder as wb  # noqa: E402


@pytest.fixture
def isolated_config(monkeypatch):
    """Stub the per-user/site stores so load_runtime_config is pure."""
    monkeypatch.setattr(model_config, '_global_runtime_store', lambda: {'sources': {}, 'model_defaults': {}})
    monkeypatch.setattr(model_config, '_global_provider_store', lambda: {})
    monkeypatch.setattr(model_config, 'load_user_model_store', lambda: {
        'narrator': {'model': 'site-model'},
        'summarizer': {'model': 'site-model'},
        'advanced_models': {
            # A user override may tune the role but never move it off Nebius.
            'world_builder': {'provider': 'site', 'temperature': 0.2},
        },
    })
    monkeypatch.setattr(model_config, 'load_site_store', lambda: {'site': {
        'baseUrl': 'https://site.example/v1', 'apiKey': 'site-key', 'api': 'openai-completions',
        'models': [{'id': 'site-model'}],
    }})
    monkeypatch.delenv('NEBIUS_BASE_URL', raising=False)
    monkeypatch.delenv('THREADLOOM_OVERRIDE_WORLD_BUILDER_MODEL', raising=False)


def test_runtime_config_has_nebius_roles(isolated_config):
    cfg = model_config.load_runtime_config()
    world_builder = cfg['models']['world_builder']
    assert world_builder['provider'] == model_config.NEBIUS_PROVIDER_NAME
    assert world_builder['model'] == 'nvidia/Nemotron-3-Ultra-550b-a55b'
    assert world_builder['temperature'] == 0.2
    assert cfg['models']['world_check']['model'] == model_config.NEMOTRON_NANO_MODEL
    assert cfg['roles']['world_builder'] == {'provider': 'llm', 'model_role': 'world_builder'}
    # The site provider is untouched.
    assert cfg['models']['narrator']['provider'] == model_config.SITE_PROVIDER_NAME


def test_nebius_provider_definition(isolated_config):
    provider = model_config.load_nebius_provider()
    assert provider['baseUrl'] == 'https://api.tokenfactory.nebius.com/v1/'
    assert provider['apiKey'] == 'env:NEBIUS_API_KEY'
    ids = [item['id'] for item in provider['models']]
    assert ids == [
        'nvidia/Nemotron-3-Ultra-550b-a55b',
        'nvidia/nemotron-3-super-120b-a12b',
        'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B',
        'nvidia/Nemotron-3_5-Lightning',
    ]


def test_nebius_provider_reads_providers_json_but_never_its_key(monkeypatch, isolated_config):
    monkeypatch.setattr(model_config, '_global_provider_store', lambda: {'providers': {'nebius-token-factory': {
        'baseUrl': 'https://api.tokenfactory.us-central1.nebius.com/v1/',
        'apiKey': 'sk-hardcoded-should-be-ignored',
        'models': [{'id': 'nvidia/Nemotron-3-Ultra-550b-a55b'}],
    }}})
    provider = model_config.load_nebius_provider()
    assert provider['baseUrl'] == 'https://api.tokenfactory.us-central1.nebius.com/v1/'
    assert provider['apiKey'] == 'env:NEBIUS_API_KEY'
    monkeypatch.setenv('NEBIUS_BASE_URL', 'https://override.example/v1')
    assert model_config.load_nebius_provider()['baseUrl'] == 'https://override.example/v1'


def test_resolve_world_builder_uses_env_key(monkeypatch, isolated_config):
    monkeypatch.setenv('NEBIUS_API_KEY', 'nb-test-key')
    resolved = model_config.resolve_provider_model('world_builder')
    assert resolved['provider_name'] == 'nebius-token-factory'
    assert resolved['provider']['baseUrl'] == 'https://api.tokenfactory.nebius.com/v1'
    assert resolved['provider']['apiKey'] == 'nb-test-key'
    assert resolved['model']['id'] == 'nvidia/Nemotron-3-Ultra-550b-a55b'
    assert resolved['timeout_seconds'] == 240
    assert resolved['response_format'] == {'type': 'json_object'}
    assert 'system' not in resolved['provider']


def test_resolve_world_builder_without_key_raises(monkeypatch, isolated_config):
    monkeypatch.delenv('NEBIUS_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='NEBIUS_API_KEY'):
        model_config.resolve_provider_model('world_builder')


def test_world_builder_llm_call_goes_to_token_factory(monkeypatch, isolated_config):
    monkeypatch.setenv('NEBIUS_API_KEY', 'nb-test-key')
    monkeypatch.setattr(llm_manager, 'load_runtime_config', model_config.load_runtime_config)
    captured = {}

    def _fake_call_model(config, system_prompt, user_prompt):
        captured['config'] = config
        return '{"sections": {}}', {'model': config['model']['id']}

    monkeypatch.setattr(wb, 'call_model', _fake_call_model)
    reply, usage = wb._call_world_builder_llm('sys', 'user')
    assert reply == '{"sections": {}}'
    cfg = captured['config']
    assert cfg['provider']['baseUrl'].startswith('https://api.tokenfactory.nebius.com/v1')
    assert cfg['provider']['apiKey'] == 'nb-test-key'
    assert cfg['model']['id'] == 'nvidia/Nemotron-3-Ultra-550b-a55b'
    assert cfg['stream'] is False
    assert usage['role'] == 'world_builder'


def test_call_model_passes_role_timeout(monkeypatch):
    import model_client

    seen = {}

    def _fake_post_json(url, payload, headers, timeout=None):
        seen.update(url=url, timeout=timeout, auth=headers.get('Authorization'))
        return {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {}}

    monkeypatch.setattr(model_client, '_post_json', _fake_post_json)
    reply, _usage = model_client.call_model({
        'provider': {'baseUrl': 'https://api.tokenfactory.nebius.com/v1/', 'apiKey': 'k', 'api': 'openai-completions'},
        'model': {'id': 'nvidia/Nemotron-3-Ultra-550b-a55b'},
        'temperature': 0.4,
        'max_output_tokens': 100,
        'stream': False,
        'timeout_seconds': 240,
    }, 's', 'u')
    assert reply == 'ok'
    assert seen == {'url': 'https://api.tokenfactory.nebius.com/v1/chat/completions', 'timeout': 240, 'auth': 'Bearer k'}


def test_resolve_world_check_disables_thinking(monkeypatch, isolated_config):
    monkeypatch.setenv('NEBIUS_API_KEY', 'nb-test-key')
    resolved = model_config.resolve_provider_model('world_check')
    assert resolved['model']['id'] == model_config.NEMOTRON_NANO_MODEL
    # With thinking on, a short budget can be spent entirely on reasoning and
    # come back as content=null (observed live on Token Factory).
    assert resolved['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}
    assert resolved['max_output_tokens'] == 400
    assert resolved['response_format'] == {'type': 'json_object'}
    # Ultra keeps reasoning on for world building.
    assert 'extra_body' not in model_config.resolve_provider_model('world_builder')


def test_call_model_merges_extra_body_without_overriding_core_fields(monkeypatch):
    import model_client

    seen = {}

    def _fake_post_json(url, payload, headers, timeout=None):
        seen['payload'] = payload
        return {
            'choices': [{'message': {'content': '{"verdict": "allowed"}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'completion_tokens_details': {'reasoning_tokens': 0}},
        }

    monkeypatch.setattr(model_client, '_post_json', _fake_post_json)
    reply, usage = model_client.call_model({
        'provider': {'baseUrl': 'https://api.tokenfactory.nebius.com/v1/', 'apiKey': 'k', 'api': 'openai-completions'},
        'model': {'id': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B'},
        'temperature': 0.1,
        'max_output_tokens': 400,
        'stream': False,
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}, 'model': 'hijack', 'max_tokens': 1},
    }, 's', 'u')
    payload = seen['payload']
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert payload['model'] == 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B'
    assert payload['max_tokens'] == 400
    assert reply == '{"verdict": "allowed"}'
    assert usage['reasoning_tokens'] == 0
