# New machine setup

This is the current Ekalavya setup path. It installs only the canonical
`ekalavya` and `eka` commands and keeps user state outside the checkout.

## 1. Checkout and prerequisites

For a public read-only installation, clone over HTTPS; no GitHub account or
SSH key is required:

```bash
git clone https://github.com/BlcakJcKd/ekalavya-delegation.git
cd ekalavya-delegation
```

Ensure Python 3.11+ and `pipx` are available. Provider clients required by
profiles you intend to use are optional and can be installed separately. Do
not put credentials in the repository.

## 2. Install

```bash
scripts/install-user-delegation.sh
command -v ekalavya
command -v eka
```

The installer is safe to rerun. Core installation/configuration failures stop
the installer; missing optional provider clients and development/test
dependencies do not. It does not alter provider authentication, model
defaults, server settings, or GPU policy.

On a fresh machine it also creates deterministic `catalogue.json` and
`profiles.json` control files from the versioned route metadata. Existing
control files are left untouched. This is bootstrap only; it does not migrate
the historical `agent-delegation` configuration or state.

## 3. Network-free validation

From a disposable directory, run:

```bash
eka --help
eka status --primary codex
eka profiles
eka models
eka history
eka doctor
```

These commands inspect local catalogue, profile, configuration, and ledger
state only. `eka status --live` is an explicit GET-only observability check
for configured shared routes; it is not an inference request.

## 4. Configuration

For interactive first-run onboarding, choose only the integrations, providers,
and model profiles you want:

```bash
eka setup
```

The UI detects local harness executables only as advisory information. It
does not install or authenticate third-party software, and Cancel writes
nothing. For an agent or non-TTY script, inspect readiness without mutation,
then use the explicit configuration commands for the user's stated choices:

```bash
eka setup --json
eka config enable-provider <provider>
eka config enable-model <profile>
```

Provider configuration, model/profile configuration, and effective
availability remain separate: disabling a provider does not erase model
choices, but its effective routes are unavailable. A selected provider with
no selected model profile is reported as a warning.

Gemini's documented zero-inference discovery path registers factual candidate
identities without changing lifecycle or the Flash default:

```bash
eka models refresh --provider gemini
```

Promotion is a separate explicit user action after inspecting `eka models`.

Inspect or explicitly change user-owned availability policy with:

```bash
eka config                 # interactive checkbox editor in a TTY
eka config --json          # deterministic inspection for scripts
eka config list --json
eka config enable-provider <provider>
eka config disable-provider <provider> --reason "maintenance"
eka config enable-model <model>
eka config disable-model <model> --reason "paused"
eka config enable <route>
eka config disable <route> --reason "paused"
```

Configuration controls eligibility, not hidden routing. Profiles and exact
model identities remain explicit, and new catalogue identities are never
automatically promoted.

## 5. Bounded execution

Use a disposable, explicitly scoped workspace and a task file:

```bash
eka run <profile> --workspace /absolute/scoped-workspace \
  --prompt-file /absolute/task.md --timeout 60 --primary <identity> --json
```

Check the resolution before execution when possible. Read the retained
response and verify it against the task. A terminal/tool yield is not a
timeout, and retries are never automatic.

## 6. Troubleshooting

Run `eka doctor` and inspect `eka status --json`. If a selected provider,
model, reasoning level, harness, workspace, or route is unavailable, fix the
declared configuration or stop. Do not weaken isolation or change shared
resource policy to force a call.
