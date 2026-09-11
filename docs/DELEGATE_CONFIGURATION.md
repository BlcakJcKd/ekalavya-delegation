# Ekalavya configuration

Configuration is user-owned availability policy. It determines which
providers, catalogue routes, and named local routes are eligible; it does not
change profile defaults or perform hidden failover.

## Routing policy

The same `config.toml` may also contain optional `[routing]` policy. Task keys
are the existing categorical slugs. Targets are canonicalized to
`primary-native`, `profile:<stable-profile>`, or `vllm:<name>`; bare stable
profile IDs are accepted by the deterministic CLI for convenience.

```toml
[routing.preferences.review]
preferred_targets = ["profile:sonnet", "profile:deepseek-flash", "primary-native"]
allowed_targets = ["profile:sonnet", "profile:deepseek-flash", "primary-native"]
excluded_targets = []

[routing.reserves.claude_weekly]
provider = "claude"
scope_kind = "account"
resource_kind = "provider_quota"
window_kind = "weekly"
minimum_remaining_fraction = 0.20
```

`preferred_targets` is ordered advice, not execution failover. A preferred
target that is unavailable may be skipped for the next recommendation target;
no delegate runs. `allowed_targets` and `excluded_targets` are hard constraints
and cannot overlap. Reserves apply only to matching fresh numeric snapshots
whose adapter/window contract and `reset_at` establish validity; otherwise
quota remains unknown.

TTY `eka setup` edits small task/preference lists. Advanced allowlists,
exclusions, and reserves use `eka config routing list`, `set-preference`,
`set-allowed`, `set-excluded`, and reserve subcommands.

For human interactive configuration, run this from a TTY:

```bash
eka config
```

It shows separate checkbox sections for providers, models, and named vLLM
routes. Changes are staged until Save; Cancel makes no changes. Provider
toggles do not rewrite model preferences: a model can remain configured
enabled while being effectively unavailable because its provider is disabled.

For inspection without a terminal UI, use the deterministic JSON form:

```bash
eka config --json
eka config list --json
```

Explicit changes use:

```bash
eka config enable <route>
eka config disable <route> --reason "maintenance"
eka config enable-model <model>
eka config disable-model <model> --reason "maintenance"
eka config enable-provider <provider>
eka config disable-provider <provider> --reason "quota policy"
```

Model and provider names are checked against Ekalavya's known catalogue. Named
vLLM routes are checked against configured local route names; typos fail
without creating new keys. When stdout is not a TTY, `eka config` emits the
same deterministic inspection payload rather than attempting an interactive
screen.

`eka config migrate` is an explicit, additive migration operation. It keeps
the source intact and is not part of ordinary discovery or status. Normal
runtime reads the active Ekalavya configuration only.

The file contains enabled flags and human-readable reasons, never credentials
or tokens. Experimental routes default to disabled until the user opts in.
Agents may inspect and respect this state, but must not mutate it merely
because a quota or resource looks low.

Configuration writes are atomic. Invalid names, sections, or values fail
without inference. `eka doctor` checks readability and ledger integrity.
