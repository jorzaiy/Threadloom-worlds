#!/usr/bin/env python3
"""Live smoke test for the Nebius Token Factory roles (needs NEBIUS_API_KEY).

Spends real tokens (a few cents per run). Two checks:

* ``import``      — import a character card end to end into a throwaway
                    character root and build ``world-bible.json`` with
                    Nemotron Ultra (role ``world_builder``). Reports import
                    blocking time, LLM latency, token usage and est. cost.
* ``world-check`` — send a few World-Check-shaped classification requests to
                    the ``world_check`` role against a short bible and report
                    verdicts and latency.

Usage (repo root)::

    python scripts/manual-checks/nebius_live_smoke.py import \
        --card examples/demo-cards/emberwell-crossing.json --out /tmp/world-bible.json
    python scripts/manual-checks/nebius_live_smoke.py world-check --bible /tmp/world-bible.json

``--env-proxy`` routes model HTTP through ``requests`` so HTTPS_PROXY is honored
(Threadloom's own client pins connections and ignores proxies by design).
Only needed in sandboxes whose egress goes through a proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / 'backend'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import model_client  # noqa: E402
from model_config import resolve_provider_model  # noqa: E402

# USD per 1M tokens (input, output) — Token Factory list prices, 2026-10.
PRICES = {
    'nvidia/Nemotron-3-Ultra-550b-a55b': (1.0, 3.0),
    'nvidia/nemotron-3-super-120b-a12b': (0.30, 0.90),
    'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B': (0.06, 0.24),
    'nvidia/Nemotron-3_5-Lightning': (0.06, 0.24),
}


def cost(model: str, usage: dict) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (usage.get('input_tokens', 0) * price_in + usage.get('output_tokens', 0) * price_out) / 1_000_000


def install_env_proxy_transport() -> None:
    import requests

    def _post_json(url, payload, headers, timeout=None):
        resp = requests.post(url, json=payload, headers=headers, timeout=model_client._request_timeout(timeout))
        if resp.status_code >= 400:
            raise model_client._build_http_error(url, resp.status_code, resp.reason, list(resp.headers.items()), resp.content)
        return resp.json()

    model_client._post_json = _post_json


def run_import(args) -> int:
    import card_importer
    import lorebook_distiller
    import world_builder
    from character_assets import clear_character_override_root, set_character_override_root

    card = json.loads(Path(args.card).read_text(encoding='utf-8'))
    captured: dict = {}
    original = world_builder._call_world_builder_llm

    def _capture(system, user):
        reply, usage = original(system, user)
        captured.update(usage)
        captured['prompt_chars'] = len(system) + len(user)
        return reply, usage

    world_builder._call_world_builder_llm = _capture
    timings: dict[str, float] = {}
    original_distill = lorebook_distiller.rebuild_lorebook_distillation

    def _timed_distill(*a, **kw):
        t0 = time.monotonic()
        try:
            return original_distill(*a, **kw)
        finally:
            timings['lorebook_distill_s'] = round(time.monotonic() - t0, 2)

    card_importer.rebuild_lorebook_distillation = _timed_distill
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'character'
        root.mkdir()
        set_character_override_root(root)
        try:
            t0 = time.monotonic()
            result = card_importer.import_card_bundle(card)
            timings['import_total_s'] = round(time.monotonic() - t0, 2)
            bible = json.loads((root / world_builder.WORLD_BIBLE_FILE).read_text(encoding='utf-8'))
        finally:
            clear_character_override_root()
    if args.out:
        Path(args.out).write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding='utf-8')
    model = captured.get('model', '')
    print(json.dumps({
        'world_bible_report': result.get('world_bible'),
        'lorebook_distillation_provider': (result.get('lorebook_distillation') or {}).get('provider'),
        'timings': timings,
        'llm_usage': {k: captured.get(k) for k in ('model', 'input_tokens', 'output_tokens', 'reasoning_tokens', 'finish_reason', 'duration_ms', 'prompt_chars', 'response_format_fallback') if k in captured},
        'est_cost_usd': round(cost(model, captured), 5),
        'setting_type': bible.get('setting_type'),
        'counts': bible.get('counts'),
        'dropped_inferred': bible.get('dropped_inferred'),
        'out': args.out,
    }, ensure_ascii=False, indent=2))
    return 0 if bible.get('status', {}).get('llm') == 'ok' else 1


DEFAULT_ACTIONS = [
    # (action, expected verdict) — expectations are for the Emberwell demo bible.
    ('I order a drink and ask Mara who has been asking about me.', 'allowed'),
    ('I pull a revolver from my coat and fire a warning shot into the ceiling.', 'out_of_world'),
    ('I prick my thumb and work a finding charm to locate the stolen mule.', 'allowed_with_cost'),
    ('I take my smartphone out and look up the ferry schedule.', 'out_of_world'),
    ('I wade into the river after dark to drag the drowned prospector ashore.', 'allowed_with_cost'),
    ('I cast a fireball at the Assay Office door.', 'out_of_world'),
    ('I slip the Ferrymen two bits to carry unassayed silver across tonight.', 'allowed_with_cost'),
    ('我把一枚银币拍在吧台上，要一张床位。', 'allowed'),
]


def run_world_check(args) -> int:
    import world_check

    bible = json.loads(Path(args.bible).read_text(encoding='utf-8'))
    if args.model:
        import os
        os.environ['THREADLOOM_OVERRIDE_WORLD_CHECK_MODEL'] = args.model
    cases = [(action, '') for action in args.action] if args.action else DEFAULT_ACTIONS
    total_cost = 0.0
    latencies = []
    mismatches = 0
    for action, expected in cases:
        result = world_check.run_world_check(action, bible=bible)
        latencies.append(result.get('latency_ms', 0) / 1000)
        total_cost += cost(result.get('model', ''), result.get('usage', {}))
        match = None if not expected else result.get('verdict') == expected
        mismatches += 1 if match is False else 0
        print(json.dumps({'action': action, 'expected': expected or None, 'match': match, 'result': result}, ensure_ascii=False))
    latencies.sort()
    print(json.dumps({
        'calls': len(latencies),
        'mismatches': mismatches,
        'latency_p50_s': round(latencies[len(latencies) // 2], 2),
        'latency_max_s': round(latencies[-1], 2),
        'est_cost_usd': round(total_cost, 5),
    }))
    return 0 if mismatches == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env-proxy', action='store_true', help='Send model HTTP through requests (honors HTTPS_PROXY).')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_import = sub.add_parser('import')
    p_import.add_argument('--card', default=str(ROOT / 'examples/demo-cards/emberwell-crossing.json'))
    p_import.add_argument('--out', default='')
    p_check = sub.add_parser('world-check')
    p_check.add_argument('--bible', required=True)
    p_check.add_argument('--model', default='', help='Override the world_check model id.')
    p_check.add_argument('--action', action='append', help='Player action (repeatable).')
    args = parser.parse_args()
    if args.env_proxy:
        install_env_proxy_transport()
    return run_import(args) if args.cmd == 'import' else run_world_check(args)


if __name__ == '__main__':
    raise SystemExit(main())
