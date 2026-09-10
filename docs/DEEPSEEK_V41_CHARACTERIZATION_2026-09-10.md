# DeepSeek V4.1 Flash characterization — 2026-09-10

## Decision

**PROMOTION_RECOMMENDED** on this small prospective screen, pending explicit
maintainer authorization. No `eka models promote` command was run. The exact
candidate is **DeepSeek V4.1 Flash**, catalogue provider model ID
`deepseek-flash`, with lifecycle `candidate` and stable Ekalavya profile ID
`deepseek-flash`.

This recommendation is limited to the frozen four-task workflow screen below;
it is not a provider-wide performance claim and does not silently change user
routing preferences.

## Provider contract confirmed

The official [DeepSeek Models & Pricing page](https://api-docs.deepseek.com/quick_start/pricing/)
reported the following on 2026-09-10:

- `deepseek-flash` is DeepSeek V4.1 Flash; `deepseek-v4-pro` remains
  DeepSeek V4 Pro / `DeepSeek-V4-Pro-0813` until the documented cutoff.
- V4.1 Flash has 1M context, 384K maximum output, thinking and non-thinking
  modes, tool calls, Responses API, Anthropic API, and provider-native vision.
- Off-peak pricing is $0.003/M cache-hit input, $0.15/M cache-miss input, and
  $0.60/M output. Peak windows are 01:00–04:00 and 06:00–10:00 UTC on
  weekdays.
- At 2026-09-14 04:00 UTC, `deepseek-v4-pro` is documented to serve V4.1
  Flash until a future exact V4.1 Pro identity exists.

The live UTC check before characterization was 2026-09-10 19:13:13 UTC, so
the four-call run was outside peak pricing and did not materially approach a
peak boundary.

## Ekalavya identity and lifecycle

- Stable profile remains `eka run deepseek-flash ...`.
- New execution uses the canonical provider endpoint `deepseek-flash`.
- Catalogue display is `DeepSeek V4.1 Flash`, generation `v4.1`, lifecycle
  `candidate`; it has no promotion basis or promotion event.
- The historical V4 identity `deepseek-v4-flash` remains a separate retired
  catalogue record named `DeepSeek V4 Flash`. It has no execution route, so a
  new run cannot execute the retired alias while labeling it V4.
- Provider aliases `deepseek-v4-flash` and
  `deepseek-v4-flash-vision-exp` are recorded as retired compatibility aliases
  and are not used for new execution.
- No V4.1 Pro catalogue entry was created. At and after the Pro cutoff,
  resolution and execution fail closed unless an exact provider-reported Pro
  identity is supplied; there is no Pro-to-Flash fallback.
- DeepSeek reasoning compatibility is preserved as
  `minimal/low -> low`, `medium/high/xhigh -> high`, and `max/ultra -> max`.
  The existing ordinary harness/profile setting remains explicitly `high`.
- Provider-native vision is recorded separately from harness capability. The
  current Codex/DeepSeek wrapper is text-only in this characterization, so
  Ekalavya does not claim multimodal execution support.

## Characterization

Run label: `deepseek-v41-flash-characterization-20260910-001`  
Crossover/task tag: `deepseek-v41-flash-characterization-20260910`  
Harness: `codex-cli 0.154.0` through `codex-deepseek`  
Requested endpoint: `deepseek-flash`  
Reasoning: `high`  
Attempts: exactly one per task; four paid calls total.

| Frozen task | Result | Wall time | Harness input tokens | Harness output tokens | Cache/reasoning/effective model | Scope violations |
|---|---:|---:|---:|---:|---|---|
| `research_python` | 5/5 | 10.52 s | 143,048 | 1,241 | unavailable / unavailable / unavailable | none |
| `diagnostic_plot` | 3/3 | 28.81 s | 389,462 | 3,565 | unavailable / unavailable / unavailable | none |
| `debug_package` | 5/5 | 14.09 s | 219,323 | 1,254 | unavailable / unavailable / unavailable | none |
| `scientific_writing` | 6/6 | 12.59 s | 143,627 | 1,431 | unavailable / unavailable / unavailable | none |
| **Total** | **19/19** | **66.01 s** | **895,460** | **7,491** | **not provider-verified** | **none** |

All four execution records show `exit_code: 0`, `execution_status:
completed`, `timed_out: false`, `retry: false`, `fallback: null`, empty
`harness_failure_reasons`, no permission denials, and required outputs
present. The automated evaluators scored all 19 checks successfully.

The Codex JSONL parser retained input/output telemetry for all four requests.
It did not expose a provider-reported model ID, cache-hit/miss split, or
reasoning-token count. Those values are null/unavailable, not inferred. Cost
is also unavailable because the run lacks a defensible provider cache split;
the official price table was not used to fabricate a cost estimate.

Underlying evidence is retained at:

`runs/deepseek-v41-flash-characterization-20260910-001/`

This ignored local evidence tree contains `run.json`, one task directory per
tag, each task's `result/execution.json` and `result/evaluation.json`, stdout,
stderr, and task deliverables. The historical August frozen portfolio and its
results were not overwritten or reinterpreted as V4.1 evidence.

## Separate operational follow-up

The previously observed usage-observability issues remain out of scope for this
lifecycle branch: Gemini/Flash status versus `harness-unavailable`, Claude
structured usage not reaching token metrics, human usage omitting JSON
success/feedback summaries, and human-output detail about external activity.
Track those on `fix/usage-observability-operational` before route-recommendation
work.
