# User installation

Ekalavya is the only supported operational delegation interface. For a public
read-only installation, use:

```bash
git clone https://github.com/BlcakJcKd/ekalavya-delegation.git
cd ekalavya-delegation
scripts/install-user-delegation.sh
```

The public clone requires no GitHub account or SSH key. Python 3.11+ and
`pipx` are required; provider CLIs are optional and are needed only for the
profiles you intend to use. For an existing checkout, run:

```bash
scripts/install-user-delegation.sh
```

The installer installs the package with pipx, creates no credentials, and
copies the canonical skill to `~/.agents/skills/delegation/SKILL.md`. If Claude
discovery exists, it keeps `~/.claude/skills/delegation` linked to that
canonical skill. Core installation/configuration failures stop the installer;
missing optional provider CLIs and development/test dependencies do not.

When both input and output are terminals, the installer offers to launch
`eka setup` after core installation. The interactive UI chooses integrations,
providers, and model profiles; it does not install models, third-party
harnesses, or provider authentication. In non-interactive use, run the
read-only readiness inspection and make only explicitly requested changes:

```bash
eka setup --json
eka config enable-provider <provider>
eka config enable-model <profile>
```

Provider selection, model/profile selection, and effective availability are
separate. Disabling a provider leaves its model choices recorded but makes
their effective availability false. A selected provider with no enabled model
profiles is reported as a configuration warning rather than silently filled.

Run repository no-model validation and the developer test suite separately:

```bash
python -m delegation.preflight
python -m unittest discover -s tests -q
```

On a fresh machine, the installer also creates deterministic
`catalogue.json` and `profiles.json` control files from the versioned route
metadata. Existing control files and user configuration are left untouched;
the installer does not migrate historical `agent-delegation` state.

## Supported commands

The package exposes exactly these project commands:

```text
ekalavya --help
eka status --primary codex
eka profiles
eka models
eka models refresh --provider gemini
eka setup
eka run <profile> --workspace DIR --prompt-file FILE
eka config
eka history
eka doctor
```

Both commands work from any current directory. `status`, `profiles`,
`models`, `config`, `history`, and `doctor` do not perform inference. Use
`eka status --live` only for an explicit GET-only shared-route observation.

## Persistent state

User-owned configuration lives under `~/.config/ekalavya/`; the private
ledger and retained execution evidence live under `~/.local/state/ekalavya/`.
The installer never overwrites existing configuration. Credentials and
provider secrets are not stored in the public repository or availability
configuration.

Profiles select stable worker capabilities. Runtime overrides are explicit:
`--provider`, `--family`, `--model`, `--reasoning`, `--harness`, and
`--timeout`. Unsupported values fail before inference; there is no silent
failover or coercion. The resolved identity is recorded in the ledger.

## Migration note

Pre-cutover delegation entry points were removed from the supported Ekalavya
installation surface. Existing historical records are retained unchanged.
