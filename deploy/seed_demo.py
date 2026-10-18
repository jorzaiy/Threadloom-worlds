"""Seed a fresh Threadloom Worlds container: Nemotron site config + demo card.

Idempotent: runs on every container start, only fills what is missing.
All roles run on NVIDIA Nemotron via Nebius Token Factory; the API key is
read from the NEBIUS_API_KEY env var (a Space/host secret), never written.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / 'backend'))

import model_config as mc  # noqa: E402
from character_manager import import_character_card_base64, list_character_cards  # noqa: E402

NARRATOR_MODEL = os.environ.get('THREADLOOM_DEMO_NARRATOR_MODEL', mc.NEMOTRON_SUPER_MODEL)
SUMMARIZER_MODEL = os.environ.get('THREADLOOM_DEMO_SUMMARIZER_MODEL', mc.NEMOTRON_SUPER_MODEL)
DEMO_CARD = APP_ROOT / 'examples' / 'demo-cards' / 'emberwell-crossing.json'


def seed_site() -> None:
    path = mc._global_site_config()
    if path.exists():
        print(f'[seed] site config exists: {path}')
        return
    models = [
        {'id': m['id'], 'name': m['name'], 'reasoning': True, 'input': ['text'], 'api': 'openai-completions'}
        for m in mc.NEBIUS_TOKEN_FACTORY_DEFAULT['models']
    ]
    site = {
        'baseUrl': os.environ.get(mc.NEBIUS_BASE_URL_ENV, mc.NEBIUS_DEFAULT_BASE_URL),
        'apiKey': f'env:{mc.NEBIUS_API_KEY_ENV}',
        'api': 'openai-completions',
        'models': models,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'site': site}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[seed] wrote site config (Nebius Token Factory): {path}')


def seed_models() -> None:
    if mc._user_model_runtime_config().exists():
        print('[seed] model config exists')
        return
    mc.update_model_config({'narrator': {'model': NARRATOR_MODEL}, 'summarizer': {'model': SUMMARIZER_MODEL}})
    print(f'[seed] narrator={NARRATOR_MODEL} summarizer={SUMMARIZER_MODEL}')


def seed_card() -> None:
    names = {str(item.get('name', '')) for item in list_character_cards()}
    if 'Emberwell Crossing' in names:
        print('[seed] demo card already imported')
        return
    payload = base64.b64encode(DEMO_CARD.read_bytes()).decode('ascii')
    result = import_character_card_base64(DEMO_CARD.name, payload, set_active=True)
    print(f"[seed] imported demo card; world bible: {json.dumps(result.get('world_bible', {}), ensure_ascii=False)[:300]}")


def main() -> int:
    if not os.environ.get(mc.NEBIUS_API_KEY_ENV):
        print('[seed] WARNING: NEBIUS_API_KEY is not set; Nemotron calls will fail until it is.')
    seed_site()
    seed_models()
    try:
        seed_card()
    except Exception as err:  # never block server start
        print(f'[seed] demo card import failed: {type(err).__name__}: {err}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
