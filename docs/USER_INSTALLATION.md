# User installation

Ekalavya is the only supported operational delegation interface. Install the
self-contained package with:

```bash
scripts/install-user-delegation.sh
```

The installer runs no-model tests and delegation preflight, installs the
package with pipx, creates no credentials, and copies the canonical skill to
`~/.agents/skills/delegation/SKILL.md`. If Claude discovery exists, it keeps
`~/.claude/skills/delegation` linked to that canonical skill.

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
