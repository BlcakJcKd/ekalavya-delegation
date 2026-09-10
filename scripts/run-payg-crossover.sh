#!/usr/bin/env bash
# Run the PAYG candidate crossover: DeepSeek V4 Pro, DeepSeek V4.1 Flash, and
# MiniMax M3 against the three initial frozen tasks (research_python,
# diagnostic_plot, debug_package), reusing existing evaluators/fixtures and
# comparing against the already-frozen tier-a-medium evidence. No historical
# contestant (Terra/Sonnet/Gemini Pro Low/Luna/Haiku/Flash) is rerun.
#
# This SPENDS REAL PAYG MONEY: up to nine paid model calls (3 candidates x 3
# tasks), one attempt each, no automatic retry. Preflight (no model calls)
# always runs first and the script refuses to proceed past a failing
# preflight.
#
# Prerequisite: `codex exec --sandbox <mode>` must actually be able to start
# a bubblewrap sandbox from this shell. Verify first, at zero cost:
#
#   bwrap --unshare-net --dev /dev --ro-bind / / /bin/true
#
# If that fails with a loopback/user-namespace error, this script's model
# calls will fail identically -- do not run it from that shell. See
# docs/PAYG_DELEGATES.md ("Smoke-test evidence") for the known failure mode
# this project hit in a nested-sandboxed Claude Code session.
#
# Usage:
#   scripts/run-payg-crossover.sh
#   scripts/run-payg-crossover.sh --run-label <run-label>
#   scripts/run-payg-crossover.sh -h | --help
#
# run-label defaults to payg-candidate-crossover-<YYYYMMDD>; a rerun with an
# existing label is refused (run labels are immutable once used).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-payg-crossover.sh [--run-label <label>]
       run-payg-crossover.sh -h | --help

Run the PAYG candidate crossover benchmark (DeepSeek V4 Pro, DeepSeek V4
Flash, MiniMax M3) against the three frozen tasks. Runs no-model checks and
preflight, then prompts for explicit confirmation before spending real PAYG
money.

Options:
  --run-label <label>  Run label to record results under (must not start
                        with '-'). Defaults to
                        payg-candidate-crossover-<YYYYMMDD>.
  -h, --help            Show this help and exit. No checks, preflight, or
                        model calls are made.
EOF
}

RUN_LABEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-label)
      if [ $# -lt 2 ]; then
        echo "error: --run-label requires a value" >&2
        exit 2
      fi
      RUN_LABEL="$2"
      shift 2
      ;;
    --run-label=*)
      RUN_LABEL="${1#--run-label=}"
      shift
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "error: unexpected argument: $1 (use --run-label <label>)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$RUN_LABEL" ] && [[ "$RUN_LABEL" == -* ]]; then
  echo "error: run label must not start with '-': $RUN_LABEL" >&2
  exit 2
fi

RUN_LABEL="${RUN_LABEL:-payg-candidate-crossover-$(date +%Y%m%d)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AGENTS="deepseek-pro,deepseek-flash,minimax-m3"
TASKS="research_python,diagnostic_plot,debug_package"
MODELS="deepseek-pro=deepseek-v4-pro,deepseek-flash=deepseek-flash,minimax-m3=MiniMax-M3"
CROSSOVER="payg-candidate-crossover"
SOURCE_TIER="tier-a-medium"

echo "== no-model checks =="
python -m unittest discover -s tests -q
python -m delegation.preflight

echo
echo "== benchmark no-model preflight (PAYG candidates) =="
python -m benchmark.runner preflight \
  --agents "$AGENTS" --tasks "$TASKS" --models "$MODELS" \
  --crossover "$CROSSOVER" --source-tier-reference "$SOURCE_TIER"

echo
echo "Preflight passed. About to spend real PAYG money: up to 9 paid model"
echo "calls (3 candidates x 3 tasks), run label: $RUN_LABEL"
read -r -p "Type 'run' to proceed, anything else to abort: " CONFIRM || CONFIRM=""
if [ "$CONFIRM" != "run" ]; then
  echo "Aborted; no model call made."
  exit 1
fi

echo
echo "== running (this spends real PAYG money) =="
python -m benchmark.runner run \
  --run-label "$RUN_LABEL" \
  --agents "$AGENTS" --tasks "$TASKS" --models "$MODELS" \
  --crossover "$CROSSOVER" --source-tier-reference "$SOURCE_TIER"

echo
echo "Results: runs/$RUN_LABEL"
echo "Remember: this does not change Ekalavya availability policy. deepseek/minimax PAYG"
echo "routes remain disabled for ordinary delegation regardless of this"
echo "benchmark's outcome -- see docs/PAYG_DELEGATES.md."
