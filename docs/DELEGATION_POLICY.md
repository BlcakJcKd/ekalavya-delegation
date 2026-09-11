# Ekalavya delegation policy

The primary agent owns routing and final correctness. Ekalavya is an explicit
control plane for bounded external work, not an autonomous router.

## Read-only recommendations

`eka route --task TASK [--primary PRIMARY]` recommends a target but never
executes it. Targets are `primary-native`, an Ekalavya profile, or a named
vLLM route. An explicit eligible user preference wins over analytical evidence;
there are no universal task/model defaults or weighted quality scores.

Same-provider-native is an execution constraint, not a universal winner. If a
Codex primary's selected work is native Codex work, the result is
`primary-native`, never `eka run terra` or `eka run luna`. A user may prefer a
cross-provider review target ahead of `primary-native`. With no task preference,
a declared primary-native target is the conservative default. Without
`--primary`, Ekalavya does not infer one and warns that native policy was not
evaluated.

## Provider rule

Use native facilities for same-provider work:

- Codex/OpenAI → native Codex agents
- Claude/Anthropic → native Claude subagents
- Gemini → native Gemini facilities

An external Ekalavya resolution that violates this returns
`same-provider-native-required`. Cross-provider execution is still explicit
and must be requested by the primary.

## Required workflow

```bash
eka status --primary <identity>
eka profiles
eka models
eka run <profile> --workspace /absolute/scoped-workspace \
  --prompt-file /absolute/task.md --timeout 60 --primary <identity>
```

Provide only bounded context in a disposable workspace. Read stdout and
retained evidence, verify the result, and reconcile disagreements. A tool
yield is not a timeout. A genuine timeout is recorded as an explicit timeout
outcome with the adapter's timeout metadata. Do not blindly retry.

Runtime overrides (`--provider`, `--family`, `--model`, `--reasoning`, and
`--harness`) are accepted only when the selected backend exposes the exact
value. No silent coercion or provider/model failover is allowed.

## Resources and configuration

Respect concurrency, quotas, credentials, and local server policy. Do not
modify shared machine, server, GPU, or provider settings to satisfy a call.
Persistent configuration is user-owned; use `eka config` only after explicit
authorization. Profiles are stable capability abstractions, while exact
provider/model identity and resolved runtime metadata are recorded per run.

## Evidence

The private ledger records requested and resolved identity, reasoning, harness,
transport, execution, tool events, usage, and cost evidence where available.
Actual, calculated, and API-equivalent costs are separate; unknown values are
null. Verify response retention and ledger recording before relying on a
delegated result.
