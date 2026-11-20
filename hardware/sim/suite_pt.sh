#!/usr/bin/env bash
# One-syndrome-per-shot bit-exactness regression (xsim, runtime tables with a table switch + RECALC in every case with >= 2 shots).
# usage: bash hardware/sim/suite_pt.sh [log]      -> one RESULT line per case
# (d 3 NS 40 nsw 128 K 2: swap-threshold refill load of the d = 11 x 40 board design)
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG=${1:-$HERE/work/suite_pt.log}
mkdir -p "$HERE/work"; : > "$LOG"
#        d NS nsw shots K B  p    seed gmin th32
for c in "3 4 8 4 1 16 0.12 5 0 29" \
         "3 6 16 4 3 14 0.12 5 0 26" \
         "3 5 32 3 2 16 0.10 7 0 29" \
         "5 8 64 6 2 16 0.12 5 4 24" \
         "5 6 16 3 1 12 0.12 3 0 29" \
         "7 16 16 3 2 16 0.15 5 0 29" \
         "7 5 32 3 1 13 0.12 5 8 29" \
         "7 8 64 4 2 16 0.12 9 8 29" \
         "11 16 8 2 2 16 0.12 5 0 29" \
         "3 32 16 3 1 16 0.12 5 0 29" \
         "3 64 16 3 1 16 0.12 5 8 29" \
         "9 16 8 2 2 16 0.12 5 0 29" \
         "13 16 4 2 2 16 0.12 5 0 29" \
         "3 4 12 4 1 16 0.12 5 0 29" \
         "5 8 20 3 2 16 0.12 5 0 29" \
         "7 8 24 3 2 16 0.12 9 0 29" \
         "5 8 22 3 2 16 0.12 3 4 24" \
         "3 40 128 4 2 16 0.12 5 0 29"; do
  set -- $c
  bash "$HERE/run_pt.sh" "$@" 2>&1 | grep -E "^RESULT" >> "$LOG" || echo "RESULT: FAIL (script) $c" >> "$LOG"
done
cat "$LOG"
