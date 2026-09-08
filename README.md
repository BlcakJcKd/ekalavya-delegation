# Ekalavya

Ekalavya is a stable delegation/control-plane abstraction for primary agents.
It separates command intent, capability profiles, model identities, explicit
resolution, execution adapters, evidence, and a private historical ledger.
The repository also contains the Ekalavya Benchmark subsystem; benchmarking
is not the product identity.

The primary agent owns routing. A profile default is an explicit choice when
the primary invokes it, not hidden autonomous provider selection. Ekalavya
does not silently fail over. Codex/OpenAI work uses native Codex agents for
same-provider Terra/Luna, Claude uses native Claude subagents for Sonnet/Haiku,
and Gemini uses native Gemini facilities. A same-provider external resolution
returns `same-provider-native-required` so the primary can decide what to do.

Profiles represent capabilities such as `coder`, `reviewer`, or
`researcher`; candidates preserve provider, family, exact provider model ID,
reasoning, harness, transport, and (when known) local serving metadata.
The live catalogue intentionally stays small: current, previous supported
fallback, and explicit new candidates. Discovery marks candidates; it never
promotes a newer model automatically, and retired models remain in history.

The private ledger at `~/.local/state/ekalavya/ledger.sqlite3` records
requested intent separately from resolved execution, benchmark/task/evaluator
hashes, request and tool telemetry, resource observations, pricing snapshots,
promotion/retirement decisions, and references to private raw evidence. API
cost, calculated cost, subscription cost, and local resource usage are kept
separate; unavailable spend is represented as `null`, never guessed.

## Commands

```text
ekalavya run <profile> [--provider P --family F --model ID --reasoning R
                        --harness H --workspace DIR --prompt-file FILE]
ekalavya status [--json]       # network-free overview
ekalavya config migrate        # additive, reversible legacy migration
ekalavya models                # catalogue only; no provider discovery
ekalavya models refresh --source FILE  # explicit, file-driven discovery
ekalavya profiles
ekalavya history [--json]
ekalavya spend [--json]
ekalavya bench                 # benchmark subsystem entry point
ekalavya doctor [--json]
```

Agent-facing instructions should use the `eka` interface:

```text
eka status --primary codex
eka profiles
eka models
eka run <profile> --workspace DIR --prompt-file FILE --primary codex
eka config                 # interactive checkbox editor in a TTY
eka config --json          # deterministic inspection for scripts
eka config enable-model <model>
eka config disable-model <model> --reason "paused"
```

`eka` is the supported shorthand installed alongside `ekalavya`. An unrelated
executable is never overwritten. Pre-cutover delegation entry points have
been removed from the Ekalavya installation surface.

`status`, `profiles`, `history`, and `spend` are read-only. The `models` list
and explicit `models refresh` path accept provider discovery data and record new
entries as candidates; it does not download models, start servers, or alter a
profile default. Persistent provider/model configuration remains user-owned.
Lifecycle promotion is a separate explicit action: inspect exact identities
with `eka models --json`, then use `eka models promote ID --basis
operational_efficiency`. Adding `--set-default --profile flash
--default-reasoning low` also changes that user-owned profile default.

## Safety and evidence

External execution adapters retain closed stdin, scoped workspaces,
symlink/parent-escape checks, recursion protection via
`AGENT_DELEGATION_DEPTH`, process-group and timeout semantics, and atomic
private response retention. Writable benchmark runs use disposable copies and
hidden evaluators outside candidate workspaces. Private configuration,
provider endpoints, credentials, raw traces, and machine-specific telemetry
must stay outside Git.

## Benchmark subsystem

The existing frozen benchmark and Benchmark V2 remain operational under
`benchmark/`. Run the no-model checks with:

```bash
python -m unittest discover -s tests -q
python -m benchmark.runner check
python -m benchmark.runner preflight --agents codex,claude,agy --tasks research_python
```

Benchmark identity includes suite/version, benchmark Git SHA, task family and
variant, content/prompt/evaluator hashes, candidate identity, harness, and
reasoning. A small longitudinal core can remain stable while evolving suites
such as Benchmark V2 provide fresh discrimination.

Ekalavya distinguishes three evaluation classes: `ordinary` delegation may
use normal project access; `public_characterization` is objective but
non-adversarially isolated, uses disposable public fixtures with visible
acceptance checks, and contains no authoritative/reference repair in a
candidate workspace; `hidden_benchmark` additionally requires proven
containment from evaluators, reference solutions, parent repositories,
sibling workspaces, and arbitrary network access. Harness eligibility is
recorded separately for each class. The public characterization suite can be
checked locally with `python -m benchmark.public_characterization.runner
check`; it does not run Benchmark V2.

## Installation

For a public read-only checkout, no GitHub account or SSH key is required:

```bash
git clone https://github.com/BlcakJcKd/ekalavya-delegation.git
cd ekalavya-delegation
scripts/install-user-delegation.sh
```

The supported user-level installation is self-contained via pipx. Python 3.11+
and `pipx` are required; provider CLIs are optional and are needed only for
the corresponding profiles. Missing optional providers are reported as
unavailable by Ekalavya and do not prevent the base installation.

```bash
scripts/install-user-delegation.sh
```

The installer does not require sudo, alter provider authentication, install a
serving engine, or modify model/GPU settings. It creates the default catalogue
and profiles on a fresh machine, without migrating historical state. It copies
the delegation skill instead of linking it into the checkout, so installed
commands remain usable after the repository moves.

Contributors with GitHub write access may use HTTPS or configure SSH for their
own checkout; public users do not need maintainer authentication.
