#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mapfile -d '' TRACKED < <(git ls-files -z)
if [ "${#TRACKED[@]}" -eq 0 ]; then
  echo "public-repo check: no tracked files found" >&2
  exit 1
fi

failures=()
for path in "${TRACKED[@]}"; do
  base="${path##*/}"
  case "$path" in
    .env|*.pem|*.key|*.p12|*.pfx|*.sqlite|*.sqlite3|*.db|\
    credentials.json|*/credentials.json|private_admin/*|*/private_admin/*|\
    delegate_runs/*|*/delegate_runs/*|.codex/*|*/.codex/*|.claude/*|*/.claude/*)
      failures+=("forbidden tracked path: $path")
      ;;
  esac
  if [[ "$base" == .env.* && "$base" != ".env.example" && "$base" != ".env.sample" ]]; then
    failures+=("forbidden tracked path: $path")
  fi
  case "$base" in
    ask-*|ask-vllm|delegate-*)
      failures+=("legacy executable path: $path")
      ;;
  esac
done

# Known provider catalogue files can contain substantial derived instruction
# material. Require an adjacent per-file notice and the repository-level
# attribution before allowing one to enter the public tree. This is a narrow
# provenance guard, not a general licence scanner.
tracked_contains() {
  local candidate="$1"
  local path
  for path in "${TRACKED[@]}"; do
    if [ "$path" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

for path in "${TRACKED[@]}"; do
  case "$path" in
    provider_templates/codex/model-catalogs/*.json)
      if grep -Fq '"instructions_template"' "$path"; then
        notice="${path%.json}.NOTICE.md"
        if ! tracked_contains "$notice" && [ ! -f "$notice" ]; then
          failures+=("catalogue missing adjacent provenance notice: $path")
        fi
        if ! grep -Fq "$path" THIRD_PARTY_NOTICES.md 2>/dev/null; then
          failures+=("catalogue missing THIRD_PARTY_NOTICES.md entry: $path")
        fi
      fi
      ;;
  esac
done

# These are high-confidence credential forms. Deliberately do not match
# ordinary words such as token, credential, password, or environment variable
# names in documentation.
secret_pattern='(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}|xox[baprs]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,}|sk-proj-[A-Za-z0-9_-]{20,}|-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)'
for path in "${TRACKED[@]}"; do
  if [ -f "$path" ] && LC_ALL=C grep -Iq . "$path" && LC_ALL=C grep -Eq "$secret_pattern" "$path"; then
    failures+=("high-confidence secret pattern in tracked file: $path")
  fi
done

python3 - <<'PY'
import sys
import tomllib
from pathlib import Path

metadata = tomllib.loads(Path("pyproject.toml").read_text())
scripts = set(metadata.get("project", {}).get("scripts", {}))
if scripts != {"eka", "ekalavya"}:
    print(f"public-repo check: unexpected package scripts: {sorted(scripts)}", file=sys.stderr)
    sys.exit(1)
PY

if ! git diff --check; then
  failures+=("whitespace error in tracked changes")
fi

if [ "${#failures[@]}" -gt 0 ]; then
  printf '%s\n' "${failures[@]}" >&2
  exit 1
fi

printf 'public-repo check: %s tracked files inspected; no high-confidence public-repository violations found\n' "${#TRACKED[@]}"
