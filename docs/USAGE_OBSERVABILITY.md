# Usage and observability (v0.4)

Ekalavya usage is local and evidence-aware. It describes only delegation
executions that Ekalavya actually observed on this machine. It is not a
complete view of a provider account: native same-provider agents, provider web
apps, other machines, and other clients are outside the local history.

## Commands

```bash
eka usage
eka usage inspect --json
eka usage --json
eka usage --period 30d --by task
eka usage export --format json --output usage.json
eka usage export --format csv --by profile
eka insights --json
eka feedback RUN_ID --outcome useful
eka feedback RUN_ID --delete
eka usage prune --before 2026-01-01T00:00:00+00:00
```

`--task` on `eka run` is explicit categorical metadata. Values are lowercase
bounded slugs (`[a-z0-9][a-z0-9._-]{0,63}`); malformed values are rejected and
the default is `unspecified`. Ekalavya never classifies prompts with another
model and never inserts the task tag into provider prompt content.

Token, latency, model identity, and cost values retain provenance. Unknown
telemetry is `null`/unavailable, not zero. Provider-reported effective model
identity is separate from the requested model and may be absent.

Hosted quota adapters are not credentialed or scraped in v0.4. Codex, Claude,
and Gemini report interactive-only status where their safe human-facing usage
views cannot be read locally; DeepSeek and MiniMax report partial capability
without inventing account headroom. Configured vLLM routes may expose local
GET-only scheduler/capacity observations, clearly scoped as `local_capacity`,
not subscription quota.

## v0.4 provider capability boundary

| Provider/harness | v0.4 status | Scope/source |
| --- | --- | --- |
| Codex/OpenAI | `interactive_only` | account usage is not read through a credentialed Ekalavya adapter |
| Claude | `interactive_only` | interactive usage commands are not treated as a machine adapter |
| Gemini/AGY | `interactive_only` | `/stats model` is not invoked as an automated quota probe |
| DeepSeek | `partial` | documented limits exist; account headroom is not locally collected |
| MiniMax | `partial` | documented limits exist; account headroom is not locally collected |
| configured vLLM | `partial`/`unavailable` | explicit GET-only local scheduler/capacity inspection, scoped as `local_capacity` |

These statuses are capability statements, not quota measurements. No provider
authentication, browser cookie, private endpoint, scraping, or model request is
used by the v0.4 quota layer.

Observability controls affect only observability-owned enrichment:
`run_observability`, explicitly owned request metrics, quota snapshots, and
current feedback. They do not delete canonical runs, resolution decisions,
promotion/default history, benchmark evidence, retained responses, catalogue
provenance, or cost observations.

All summaries are deterministic and computed in UTC. Periods use half-open
`[start_utc, end_utc)` boundaries. Median uses sorted linear
interpolation at `(n-1)*0.5`; P90 uses sorted linear interpolation at
`(n-1)*0.9` and is shown only for at least ten observations. Small samples are
descriptive and do not establish model quality or cross-provider efficiency.
