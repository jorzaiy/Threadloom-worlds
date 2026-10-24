# Threadloom Worlds

**A long-form role-play runtime that builds a world bible from any character card and keeps the story inside that world, powered by NVIDIA Nemotron on Nebius Token Factory and Tavily.**

[中文说明](README.zh-CN.md) · License: MIT · Live demo: https://threadloom-worlds.onrender.com/ (free tier, first load may take ~1 min to wake)

## The problem

Character cards for AI role-play carry a whole world: places, factions, history, laws. Chat models forget it within a few dozen turns. The cafe appears in a desert town, a forbidden act goes unpunished, a medieval guard pulls out a phone. Players call this "drift".

## What Threadloom Worlds does

1. **World Builder** – when a card is imported, every setting the card states is extracted into a structured **world bible** (geography, factions, social system, tech/supernatural limits, economy, customs, taboos, era & feel). Long lore entries are split, never truncated; character personas and author instructions are filtered out. Nemotron Ultra then fills only the gaps, tagged `inferred`, and an inferred entry can never override the card.
2. **Grounded research (Tavily)** – a Nemotron planner decides whether the setting is *historical*, *analog* (e.g. a wuxia world that feels Song/Ming) or *pure fantasy*. For the first two it writes historian-style queries, runs Tavily advanced search, and the builder cites sources `R1..R8` on each inferred entry. Fantasy cards skip the search.
3. **World Skeleton** – a compact, location-aware slice of the bible (world-wide rules plus entries for the current region) is injected into every narrator prompt, so the base layout of the world never falls out of context.
4. **World Check** – every player turn, Nemotron Super classifies the action as *allowed*, *allowed with a cost*, or *out of world*, names the rule that fired, and hands the narrator a concrete hint. The UI shows the verdict on each reply.
5. **Editable canon** – players can view, edit, lock or rebuild the bible. Locked entries are canon for both the narrator and World Check.

## How NVIDIA Nemotron and Nebius Token Factory are used

All model calls go through Nebius Token Factory's OpenAI-compatible API (`https://api.tokenfactory.nebius.com/v1/`, key from `NEBIUS_API_KEY`), provider `nebius-token-factory` in `backend/model_config.py`.

| Role | Model | Where |
|---|---|---|
| World Builder (bible inference) | `nvidia/Nemotron-3-Ultra-550b-a55b` | `backend/world_builder.py` |
| Research planner (no-think) | Nemotron Super 120B | `backend/world_builder.py` |
| World Check (per turn) | Nemotron Super 120B | `backend/world_check.py` |
| Narrator / summarizer in the demo | Nemotron Super 120B | `deploy/seed_demo.py` |

A full bible build costs about $0.01–0.02; a World Check is a fraction of a cent.

## How Tavily is used

`backend/world_builder.py` calls the Tavily Search API at runtime (`TAVILY_API_KEY`, advanced depth, era-term and domain filtering) during bible builds for historical or analog settings. Results become numbered notes `R1..R8`, and the UI links each bible entry to the sources it cites.

## Run it

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=...        # Nebius Token Factory
export TAVILY_API_KEY=...        # optional, enables grounded research
cd backend && ./start.sh          # http://127.0.0.1:8765
```

Import a card from **Settings → 导入资产**, or use the bundled demo card `examples/demo-cards/emberwell-crossing.json`. The **总结面板 → World** section shows the bible and the last World Check.

### Deploy (Render, free tier)

`render.yaml` is a Blueprint: Docker build from `deploy/huggingface/Dockerfile`, seeds the demo card on start. Set `NEBIUS_API_KEY`, `TAVILY_API_KEY` and `THREADLOOM_ADMIN_PASSWORD` in the dashboard.

### Tests

```bash
env -u NEBIUS_API_KEY -u TAVILY_API_KEY python -m pytest -q tests
```

## What was built during the hackathon

Threadloom (the base RP runtime: memory layers, state keeper, multi-session UI) existed before the hackathon. Everything below is new in this repo, on top of the base commit `b39b3f6`:

- Nebius Token Factory provider and Nemotron role routing
- World Builder, card lore extraction and the world bible format
- Tavily research planner and cited sources
- World Skeleton injection into the narrator
- Per-turn World Check and its UI
- World bible view / edit / lock / rebuild API and panel
- Render deployment kit and demo seeding
