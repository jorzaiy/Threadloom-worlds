# Manual checks

These are **live, end-to-end scripts**, not pytest unit tests. They drive a
running backend (real `handle_message` / a real HTTP server / real character
data) and were previously parked under `tests/` where pytest tried — and failed
— to collect them:

- they take required positional arguments (`session_id`, …), so pytest cannot
  collect them as test functions, and
- `http_regression_check.py` imported `backend.*` at module top level, which
  raised `ModuleNotFoundError` during collection depending on test order — a
  latent landmine that masked itself in full runs.

They were moved out of `tests/` (and the `test_` prefix dropped) so the pytest
suite stays clean and order-independent. Run them by hand against a live server
when you want a full smoke pass:

```bash
# from the repo root, with the backend running and a character/session ready
python3 scripts/manual-checks/http_regression_check.py --session <session_id>
python3 scripts/manual-checks/keeper_e2e_check.py <session_id>
python3 scripts/manual-checks/keeper_summary_check.py <session_id>
python3 scripts/manual-checks/full_regression_check.py <session_id>
```

If any of these grow deterministic, LLM-stubbed assertions, promote that part
back into `tests/` as a real unit test instead.

## Nebius Token Factory live smoke (`nebius_live_smoke.py`)

Needs `NEBIUS_API_KEY`; spends real tokens (about $0.01 per import run, well
about $0.0006 per World Check call on Super). No running server needed.

```bash
# Import the demo card end to end and build world-bible.json with Nemotron Ultra
python3 scripts/manual-checks/nebius_live_smoke.py import \
    --card examples/demo-cards/emberwell-crossing.json --out /tmp/world-bible.json
# World-Check-shaped classification calls against that bible (role world_check)
python3 scripts/manual-checks/nebius_live_smoke.py world-check --bible /tmp/world-bible.json \
    [--model nvidia/Nemotron-3_5-Lightning]
```

Add `--env-proxy` (before the subcommand) when egress must go through
`HTTPS_PROXY`; Threadloom's own HTTP client pins connections and ignores proxies.
