# Ekalavya

[![CI](https://github.com/BlcakJcKd/ekalavya-delegation/actions/workflows/ci.yml/badge.svg)](https://github.com/BlcakJcKd/ekalavya-delegation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version: 0.4.0](https://img.shields.io/badge/version-0.4.0-orange.svg)](CHANGELOG.md)

Ekalavya's original source is licensed under the [MIT License](LICENSE).
Third-party or derived material is distributed under the licences identified
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Ekalavya is a local control plane for deliberate AI delegation. It lets a
primary coding agent select a stable profile, resolve an exact available
model identity, and keep an auditable private record of what happened. It
does not silently fail over between providers.

Use it when you want one small, explicit interface for the integrations you
actually use—without putting credentials, provider defaults, or history in a
Git checkout.

## Quick install

Core requirements: Python 3.11+ and [pipx](https://pipx.pypa.io/). A GitHub
account or SSH key is not required for a public read-only install.

```bash
git clone https://github.com/BlcakJcKd/ekalavya-delegation.git
cd ekalavya-delegation
scripts/install-user-delegation.sh
```

The user-level installer creates private Ekalavya configuration, catalogue,
and profile control files when absent. It is safe to rerun and never migrates
or replaces existing user configuration or history. It does not use `sudo`,
change system Python, install provider software, or request credentials.

## Choose integrations and model profiles

Run the first-run setup UI in a terminal:

```bash
eka setup
```

Choose only the integrations, providers, and model profiles you want. Detection
is advisory: a selected integration can remain selected while its provider-owned
authentication or CLI is still incomplete. Unselected integrations need no
credentials and do not block core Ekalavya use. Run `eka setup` again later to
adjust selections; Cancel makes no changes.

For scripts and AI coding agents, `eka setup --json` is a read-only readiness
report. Use the explicit `eka config` commands to make only the selections the
user requested.

## First delegation

Inspect the local setup first:

```bash
eka doctor
eka status --primary codex
eka profiles
```

Then delegate with an explicit, scoped task. This command may invoke the
selected provider:

```bash
eka run flash --workspace /absolute/project --prompt-file /absolute/task.md --primary codex
```

## Common commands

```bash
eka setup                         # interactive integration/profile selection
eka setup --json                  # read-only readiness for automation
eka config                        # edit availability in a TTY
eka config --json                 # inspect effective availability
eka models                        # inspect the local model catalogue
eka models refresh --provider gemini  # zero-inference Gemini discovery
eka models promote ID --basis manual  # explicit lifecycle promotion
eka profiles                      # stable profile definitions
eka doctor                        # local installation/configuration checks
```

Discovery registers factual candidates; it never promotes a generation or
changes a profile default. Promotion is always an explicit, user-owned action.

## Integrations

Ekalavya currently has profiles for Codex/OpenAI (Terra, Luna), Claude
(Haiku, Sonnet), Gemini Flash, DeepSeek V4 Flash/Pro, MiniMax M3, and named
local/vLLM routes where configured. Each integration is optional: install and
authenticate only its own provider/harness tools using that provider's normal
flow. Missing optional integrations are reported as unavailable rather than
making core setup fail.

## Install with an AI coding agent

See [the agent installation guide](docs/AGENT_INSTALLATION.md). A concise
prompt to copy is:

```text
Install Ekalavya from https://github.com/BlcakJcKd/ekalavya-delegation.
Read README.md and docs/AGENT_INSTALLATION.md first. Use the supported
user-level installer; do not use sudo or modify system Python. Preserve any
existing Ekalavya configuration and state. Do not configure providers I did
not select or ask me to paste API keys. Interactive humans may use `eka setup`;
non-TTY agents/scripts must not drive the interactive UI. Inspect with the
read-only `eka setup --json`, apply requested deterministic changes with the
existing `eka config` commands, then finish with `eka doctor` and report which
selected integrations are ready. Do not invoke models merely to verify setup.
```

## More detail

The concise install reference is in [docs/USER_INSTALLATION.md](docs/USER_INSTALLATION.md).
For a new machine, see [docs/NEW_MACHINE_SETUP.md](docs/NEW_MACHINE_SETUP.md).
Control-plane, lifecycle, privacy, and evidence design is documented in
[docs/EKALAVYA_CONTROL_PLANE.md](docs/EKALAVYA_CONTROL_PLANE.md); benchmark
material remains in [docs/AGENT_BENCHMARK_HANDBOOK.md](docs/AGENT_BENCHMARK_HANDBOOK.md).
Contributors may use HTTPS or their own GitHub SSH setup; ordinary public
users need neither write access nor SSH credentials.
