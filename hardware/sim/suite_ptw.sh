#!/usr/bin/env bash
# Real-time bit-exactness regression (xsim vs nimbus.decoding.real_time with the hardware schedule): every sweep record of every
# replica and every commit record of all windows (two open windows and the final one per shot).
# usage: bash hardware/sim/suite_ptw.sh [log]      -> one RESULT line per case
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG=${1:-$HERE/work/suite_ptw.log}
mkdir -p "$HERE/work"; : > "$LOG"
#        d R  k lram nsw shots K p    seed rs srl carry [cs]   (p < 1/28: rank mode 1; carry 1: seeded once per shot; cs 1: C-slow loop; swp 1: registered swap decision)
for c in "3 4 1 0 8 2 1 0.04 5 1 0 0" \
         "3 8 1 0 64 3 2 0.05 7 1 0 1" \
         "3 16 1 0 32 2 1 0.02 3 1 1 1" \
         "5 4 2 0 16 2 1 0.04 5 1 0 1" \
         "5 8 2 0 32 2 2 0.03 9 1 1 1" \
         "7 4 5 1 8 1 1 0.04 5 1 0 1" \
         "7 6 5 1 32 2 2 0.03 11 2 0 1" \
         "7 4 4 0 8 1 1 0.04 5 1 0 1" \
         "9 4 9 1 4 1 1 0.04 5 2 0 1" \
         "9 8 9 1 8 2 2 0.03 7 2 1 1" \
         "7 6 5 1 8 1 2 0.04 5 3 0 1" \
         "9 6 9 1 4 1 2 0.04 3 3 0 1" \
         "7 6 5 1 8 1 1 0.04 5 3 0 1 1" \
         "9 6 9 1 8 2 1 0.03 7 3 0 1 1" \
         "9 6 9 1 4 1 2 0.02 3 3 0 1 1" \
         "3 8 1 0 32 2 1 0.03 5 1 0 1 0 1" \
         "9 6 9 1 8 1 1 0.04 9 3 0 1 1 1" \
         "11 4 11 1 4 1 1 0.04 3 4 0 1 1 1" \
         "11 12 11 1 4 1 1 0.03 5 6 0 1 1 0" \
         "11 12 11 1 4 1 2 0.03 5 6 0 1 1 1"; do
  set -- $c
  bash "$HERE/run_ptw.sh" "$@" 2>&1 | grep -E "^RESULT" >> "$LOG" || echo "RESULT: FAIL (script) $c" >> "$LOG"
done
cat "$LOG"
