# Evaluation and regression suite (M11, spec FR-13 / §19)

The suite answers one question before any change ships: **is the diagnostic pipeline at
least as good as it was?** It is the spec's release gate made executable:

| Spec §19 gate | How the suite checks it |
|---|---|
| Data quality | every locked case's quality verdict (PASS / DEGRADED / BLOCKED) must match |
| ADAS metric thresholds | locked pass/fail verdicts per metric, plus metrics vs the generator's own ground truth |
| LLM evidence gate | evidence support rate ≥ 90 %, unsupported-claim rate ≤ 10 % (defaults, configurable) |
| Regression on a locked scenario suite | compared with `evals/baseline.json`: no aggregate worse, no case that passed now failing |
| p50 / p95 latency, tokens | recorded per run and aggregated |

Everything is synthetic. The numbers measure the pipeline and the agent, never real-world
AEB performance; every report says so.

## Layout

```
backend/app/eval/
  schemas.py   EvalCase, CaseResult, Aggregates, GateThresholds, EvalReport (Pydantic)
  suite.py     LOCKED_SUITE (10 cases) + suite_sha256()   <- the only file that defines expectations
  scoring.py   pure scoring: deterministic checks, agent checks, aggregate(), apply_gate()
  runner.py    run_case / run_suite, JSON + Markdown writers
  cli.py       python -m app.eval.cli {list, run}
backend/tests/test_eval.py
evals/baseline.json      the promoted live baseline (tracked)
evals/results/           every live run (gitignored)   evals/ci/  CI output (gitignored)
```

## The locked suite (`aeb-locked-v1`)

| case | expected | what is pinned |
|---|---|---|
| `late_braking_seed42` / `seed7` / `seed2026` | fault | PASS quality; events TTC-crossing, brake, collision; latency, collision, min-TTC, dropout, min-confidence all FAIL; top cause ∈ {perception_error, fusion_error} |
| `late_braking_near_miss` | fault | 0.9 s confidence collapse: latency 0.52 s FAIL, no collision; same acceptable causes |
| `nominal_seed42` / `seed7` / `nominal_60kmh` | nominal | all locked metrics PASS, collision event forbidden; **no fault hypothesis at medium or higher** |
| `blocked_missing_ego_speed` | blocked | `required_signals` gate must refuse the file before any analysis |
| `degraded_confidence_nan_burst` | fault | DEGRADED quality; fault still found; report confidence capped at medium |
| `degraded_timestamp_gap_nominal` | nominal | DEGRADED quality; no false fault; cap at medium |

Expectations come from the generator's ground truth (`ScenarioMetadata.ground_truth`),
never from the metrics code, otherwise a regression could re-pin itself. Two extra checks
use that ground truth directly as an independent oracle: the collision flag must agree and
the braking-latency metric must be within 0.05 s (two samples at 50 Hz) of the generator's
value.

Why perception/fusion and not planner: the late-braking generator delays detection by
collapsing object confidence during the risk phase. Once the target is valid again the
planner brakes within the nominal 0.16 s. Blaming the planner is therefore the wrong root
cause, which is exactly what the live model tends to do (see baseline).

`suite_sha256()` hashes the case definitions (stable across Python hash seeds). Results and
baselines carry it; a baseline from a different suite is refused with "re-baseline first".
Edit the suite deliberately, then promote a fresh baseline in the same PR.

## Running it

```
# deterministic checks only, no model (this is the CI step)
$env:LLM_PROVIDER="fake"; .venv\Scripts\python.exe -m app.eval.cli run --no-agent --out ../evals/ci

# live: agent + verifier on every non-blocked case, compared with the baseline
.venv\Scripts\python.exe -m app.eval.cli run --index ../data/index_openai --access internal `
    --out ../evals/results --baseline ../evals/baseline.json
```

Exit codes: 0 gate passed, 1 gate failed, 2 unknown case id, 3 agent requested with the
fake provider (it refuses rather than pretend). `--cases a,b` runs a subset; a subset is
compared with the baseline per case only, because aggregates over different case sets are
not comparable. `--min-top1` / `--max-false-fault` add absolute accuracy bars on top of the
baseline rule.

Each run writes `evals/results/<eval_id>.json` (the full record: every check, every run and
verification id, prompt hash, tokens, latency) and a Markdown summary next to it.

### Promoting a baseline

A baseline is just a result you have decided to defend:

```
copy evals\results\eval_<id>.json evals\baseline.json
```

Do it when a change is meant to improve the agent and the new run shows it. Never promote a
worse run to make the gate green.

## Baseline (2026-11-10, gpt-4o-mini-2024-07-18, real embeddings)

See `evals/baseline.json`; the numbers below are copied from it.

| run | cases pass/fail | top-1 cause accuracy (5 fault cases) | false-fault rate (4 nominal cases) | evidence support | unsupported claims | p50 / p95 latency | tokens in / out |
|---|---|---|---|---|---|---|---|
| `eval_b952ec9fe004` **(baseline)** | 3 / 7 | 0 % (0/5; the right cause was ranked 2nd every time) | 50 % (2/4) | 100 % | 0 % | 12.0 s / 23.8 s | 31 682 / 5 576 |
| first live run, same code, 20 min earlier (not comparable by hash: it predates the hash fix) | 3 / 7 | 40 % (2/5) | 100 % (4/4) | 100 % | 0 % | 8.1 s / 15.3 s | 25 509 / 4 628 |
| two-case re-check against the baseline | 1 / 1 | n/a | 100 % (1/1) | 100 % | 0 % | 8.0 s | 2 441 / 404 |

**Run-to-run variance is large.** With no code change, `nominal_seed42` passed in the
baseline and failed in the re-check minutes later, and top-1 accuracy swung between 0 % and
40 %. Nine agent cases sampled once each cannot separate a real regression from sampling
noise, so a single-run baseline comparison will flap. The agent already requests
`temperature=0.0, seed=0`; the variance is the provider's own non-determinism, so the fix
is repeated sampling per case (run each agent case k times and score the pass rate), which
is the next step for the suite. Until then, treat a per-case "passed before, fails now"
failure as a prompt to look, not as proof of a regression.

What the baseline says: the evidence discipline works (every cited ID resolves, nothing
unsupported survives the verifier), but the agent's *reasoning* has two measured defects:

1. **Wrong top cause** on most collision cases: it ranks `planner_logic_error` first and
   the perception-confidence collapse second, although the planner reacted within spec once
   the target was valid.
2. **False faults on every nominal log**: it reports medium-confidence planner or control
   problems on logs where every metric passes. The default question ("Why did the AEB brake
   late…?") presupposes a failure, and the prompt has no instruction for the no-violation
   case.

Both are now regression-tested: a prompt change that fixes them must raise top-1 accuracy
and lower the false-fault rate against this baseline, and a change that breaks them again
fails the gate.
