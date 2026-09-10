#!/usr/bin/env bash
# Run the PAYG stage-2 scientific-writing screen: DeepSeek V4 Pro, DeepSeek
# V4 Flash, and MiniMax M3 against the single frozen `scientific_writing`
# task, reusing existing evaluators/fixtures. This is the smallest approved
# second-stage PAYG experiment (see docs/PAYG_BENCHMARK_2026-08.md, "Proposed
# second-stage benchmark") -- it does not run research_python,
# diagnostic_plot, debug_package, pandoc_pdf, or any repository_review
# variant, and it does not rerun any historical (Terra/Sonnet/Gemini Pro
# Low/Luna/Haiku/Flash) contestant.
#
# This SPENDS REAL PAYG MONEY: up to three paid model calls (one per
# candidate), one attempt each, no automatic retry. Preflight (no model
# calls) always runs first and the script refuses to proceed past a
# failing preflight.
#
# Prerequisite: same bwrap sandbox prerequisite as the stage-1 launcher; see
# scripts/run-payg-crossover.sh and docs/NEW_MACHINE_SETUP.md.
#
# Usage:
#   scripts/run-payg-stage2-writing.sh
#   scripts/run-payg-stage2-writing.sh --run-label <run-label>
#   scripts/run-payg-stage2-writing.sh -h | --help
#
# run-label defaults to payg-stage2-writing-<YYYYMMDD>; a rerun with an
# existing label is refused (run labels are immutable once used).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-payg-stage2-writing.sh [--run-label <label>]
       run-payg-stage2-writing.sh -h | --help

Run the PAYG stage-2 scientific-writing screen (DeepSeek V4 Pro, DeepSeek
V4 Flash, MiniMax M3) against the single frozen `scientific_writing` task.
Runs no-model checks and preflight, then prompts for explicit confirmation
before spending real PAYG money.

Options:
  --run-label <label>  Run label to record results under (must not start
                        with '-'). Defaults to
                        payg-stage2-writing-<YYYYMMDD>.
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

RUN_LABEL="${RUN_LABEL:-payg-stage2-writing-$(date +%Y%m%d)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AGENTS="deepseek-pro,deepseek-flash,minimax-m3"
TASKS="scientific_writing"
MODELS="deepseek-pro=deepseek-v4-pro,deepseek-flash=deepseek-flash,minimax-m3=MiniMax-M3"
CROSSOVER="payg-stage2-writing"
SOURCE_TIER="tier-a-medium"

echo "== no-model checks =="
python -m unittest discover -s tests -q
python -m delegation.preflight

echo
echo "== benchmark no-model preflight (PAYG stage 2: scientific_writing) =="
python -m benchmark.runner preflight \
  --agents "$AGENTS" --tasks "$TASKS" --models "$MODELS" \
  --crossover "$CROSSOVER" --source-tier-reference "$SOURCE_TIER"

echo
echo "Preflight passed. About to make at most 3 PAYG model calls:"
echo "    DeepSeek V4 Pro"
echo "    DeepSeek V4 Flash"
echo "    MiniMax M3"
echo
echo "Task:"
echo "    scientific_writing"
echo
echo "Run:"
echo "    $RUN_LABEL"
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
