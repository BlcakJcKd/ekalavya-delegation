# Contributing to Ekalavya

Ekalavya is a public, local control plane for deliberate delegation. Changes
should preserve explicit user choice, provider routing boundaries, private
user-owned configuration and state, and the rule that inspection and tests do
not invoke models.

Ekalavya's original source is licensed under MIT. Check
`THIRD_PARTY_NOTICES.md` before adding or adapting third-party or derived
material, and keep the applicable notices with any distributed copy.

Before opening a pull request:

1. Create a focused branch from `main`.
2. Install the test dependencies with `python -m pip install -e '.[test]'`.
3. Run `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q`.
4. Run `python -m benchmark.v2.runner check` and
   `python -m benchmark.runner check` when changing benchmark or packaging
   behavior.
5. Run `scripts/check-public-repo.sh` and inspect the complete diff.

Do not add provider credentials, private experiment state, generated caches,
or model calls to a pull request. Keep public package entry points limited to
`eka` and `ekalavya`; legacy `ask-*`, `ask-vllm`, and `delegate-*` commands
are not part of the public interface.
