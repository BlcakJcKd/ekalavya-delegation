# Installing Ekalavya with a coding agent

This guide is for Codex, Claude Code, OpenCode, and similar agents installing
Ekalavya on behalf of a user. It is an installation guide, not contributor
remote setup or provider credential documentation.

## Safe procedure

1. Clone the public repository with HTTPS:

   ```bash
   git clone https://github.com/BlcakJcKd/ekalavya-delegation.git
   cd ekalavya-delegation
   ```

2. Inspect an existing checkout and `~/.config/ekalavya/` before changing
   anything. Preserve existing `config.toml`, catalogue, profiles, and
   `~/.local/state/ekalavya/` history. A clean machine needs no legacy
   migration.

3. Confirm the core prerequisites established by the project: Python 3.11+
   and `pipx`. Run the supported user installer:

   ```bash
   scripts/install-user-delegation.sh
   ```

   Never use `sudo`, install globally, or alter system Python. The installer
   bootstraps versioned default control files when absent, installs only
   `eka` and `ekalavya`, installs the delegation skill, and preserves existing
   user-owned state.

4. For a person at a terminal, run `eka setup` and let them choose the
   integrations, providers, and model profiles they want. The setup screen
   must not install third-party harnesses or authenticate them.

   For non-interactive automation, inspect only with:

   ```bash
   eka setup --json
   ```

   Then use existing deterministic `eka config enable-provider`,
   `eka config disable-provider`, `eka config enable-model`, and
   `eka config disable-model` commands for the choices the user explicitly
   made. Do not drive a curses interface and do not invent a second setup API.

5. Provider CLIs and provider authentication are optional and provider-owned.
   Do not configure unselected providers, create or request raw secrets, or
   claim authentication merely because a binary is present. Report clear
   provider-owned next actions when selected integrations are not ready.

6. Verify without model calls:

   ```bash
   eka --help
   ekalavya --help
   eka setup --json
   eka doctor
   eka config --json
   eka profiles
   eka models
   ```

   Do not invoke a model merely to verify installation. Report which selected
   integrations are ready and which optional prerequisites remain unresolved.

## Boundaries

Do not copy configuration from another machine, alter provider defaults,
delete historical state, or restore legacy commands. Keep credentials, SSH
material, private endpoints, raw provider traces, and local configuration out
of Git and out of chat output. Public users should use the HTTPS clone URL;
maintainers may independently choose HTTPS or SSH for their own write access.
