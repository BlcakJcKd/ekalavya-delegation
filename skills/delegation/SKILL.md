---
name: delegation
description: >
  Teaches agents how to use the Ekalavya delegation control plane safely,
  including profile resolution, runtime selection, evidence, and ledger
  verification.
---

# Ekalavya delegation

Ekalavya is an explicit delegation control plane. It is optional capacity, not
an obligation. The primary agent remains responsible for routing, correctness,
scope, and the final decision.

## Start with state

Use the canonical Ekalavya commands before considering an external call:

```bash
eka status --primary codex
eka profiles
eka models
eka history
```

Use `eka doctor` for a health check and `eka config --json` to inspect
persistent configuration from an agent or script. For a human in a TTY,
`eka config` opens the interactive checkbox editor with Save/Cancel. These
commands are safe to run from any working directory and do not perform
inference. `eka models refresh` is explicit and discovery-only;
it records new identities as candidates and never promotes them or changes a
profile default.

Lifecycle promotion is a separate explicit action. Inspect exact identities
with `eka models --json`; a promotion based on measured efficiency must use
`eka models promote ID --basis operational_efficiency`. Adding
`--set-default --profile flash --default-reasoning low` also changes that
user-owned profile default, so it must not be run implicitly.

## Primary owns routing

The primary decides whether to delegate, how much context to provide, and to
whom. `eka run <profile>` explicitly selects that profile's configured
default. Ekalavya has no silent provider or model failover. An unavailable,
ambiguous, or invalid resolution is an actionable result, not permission to
try a different provider implicitly.

## Same-provider native rule

Same-provider work belongs to the provider's native agent facility:

- Codex/OpenAI → native Codex agents
- Claude/Anthropic → native Claude subagents
- Gemini → native Gemini facilities

When an external Ekalavya execution would violate this rule, resolution must
return `same-provider-native-required`. The primary decides whether to use its
native facility or stop. Cross-provider Ekalavya execution remains explicit.

## Runtime selection

When the selected backend exposes them, `eka run` supports explicit overrides:

```bash
eka run <profile> --provider <provider> --family <family> \
  --model <provider-model-id> --reasoning <setting> --harness <harness> \
  --workspace /absolute/scoped-workspace \
  --prompt-file /absolute/task.md --primary <primary-identity>
```

Only values actually supported by the selected backend may be used.
Unsupported provider, family, model, reasoning, or harness values must fail
before inference. Never silently coerce a requested value. The ledger records
both the requested value and the exact resolved value.

## Profiles and model lifecycle

Profiles are stable worker/capability abstractions, not aliases that promise a
permanent provider model. Exact provider and model IDs are resolved at
execution time and recorded. Catalogue identities may be `current`, `previous`
(supported fallback), or `candidate`. New discovery creates candidates only;
new models are never automatically promoted. Historical identities remain
available for audit even when they are no longer selectable.

## External delegation contract

For an external call:

1. Check status and profile resolution first.
2. Provide only bounded context in a disposable, declared workspace.
3. Use an explicit timeout appropriate to the task.
4. Read stdout and the retained response evidence.
5. Verify the result against the task and workspace contract.
6. Reconcile disagreements with other evidence; the primary owns final
   correctness.

A terminal or tool yield is not a delegate timeout. A timeout is an explicit
execution outcome. Do not blindly retry: preserve the first result and
determine whether a retry is justified by the task and evidence.

## Shared and local resources

Respect backend-specific concurrency, quota, and resource state. Do not change
machine, server, GPU, provider, or harness policy merely to satisfy a
delegation. If the selected backend cannot meet the isolation or selection
contract, report it as unavailable rather than weakening the contract.

## User-owned configuration

Persistent Ekalavya configuration is user-owned. Agents may read and respect
it, and may use an explicitly requested one-shot override. Do not permanently
change profiles, catalogue lifecycle, provider settings, defaults, or resource
policy unless the user explicitly instructs that change.

Explicit configuration mutations use deterministic commands such as
`eka config disable-provider <provider> --reason "..."`,
`eka config enable-provider <provider>`, `eka config disable-model <model>`,
and `eka config enable-model <model>`. Provider availability and model
configuration are separate: a configured-enabled model is effectively
unavailable while its provider is disabled. Unknown names fail validation;
they are never created implicitly.

## Ledger and evidence

Runs are recorded in the private Ekalavya ledger where supported. Requested
intent, resolved provider/model identity, reasoning, harness and version,
adapter/transport, benchmark identity, execution telemetry, tool events,
resource observations, and cost evidence are distinct fields. Missing token, cost, latency, or resource values remain null; never invent them. Actual,
calculated, and API-equivalent cost must remain separately labelled.

For a completed call, verify the response-retention metadata and the ledger
entry before relying on the result. A retained response is evidence of what was
returned, not proof that the answer is correct.

## Usage observability

Pass an explicit categorical task tag when invoking Ekalavya:

```bash
eka run <profile> --task review --workspace ... --prompt-file ...
```

Task values are lowercase bounded slugs (`[a-z0-9][a-z0-9._-]{0,63}`), or
`unspecified` when omitted. Do not infer a task from prompt text and do not
ask another model to classify it. The tag is metadata only; it does not route,
change defaults, change reasoning, or alter provider availability.

`eka usage` reports the local Ekalavya-observed history and separately labels
honest provider headroom/capability states. Native same-provider subagents and
activity from provider applications, other clients, or other machines are not
automatically visible. `eka insights` is deterministic local aggregation and
must not make model-quality claims from tiny samples. `eka feedback RUN_ID
--outcome useful|mixed|not-useful` records one replaceable local outcome only;
feedback never changes routing or defaults.
