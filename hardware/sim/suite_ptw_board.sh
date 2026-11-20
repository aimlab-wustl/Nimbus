#!/usr/bin/env bash
# Real-time board-level regression (xsim): the driver's session (timed batch, throughput batch with new tables and RECALC,
# reseeded tight batch) replayed through AXI4-Lite on ptw_board_top, two clocks.
# usage: bash hardware/sim/suite_ptw_board.sh [log]
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG=${1:-$HERE/work/suite_ptw_board.log}
mkdir -p "$HERE/work"; : > "$LOG"
#        d R  k lram rs cs nsw shots K p    seed srl    (the chosen layouts at reduced R / nsw; d = 3 at the chosen R = 16)
for c in "3 4 1 0 1 0 8 4 1 0.04 5 0" \
         "3 16 1 0 1 0 16 3 1 0.04 7 1" \
         "5 4 2 0 1 0 8 3 1 0.04 5 0" \
         "7 4 5 1 1 0 8 2 1 0.04 5 0" \
         "9 6 9 1 3 1 4 2 1 0.04 3 0" \
         "11 4 11 1 4 1 4 2 1 0.04 3 0"; do
  set -- $c
  bash "$HERE/run_ptw_board.sh" "$@" 2>&1 | grep -E "^RESULT" >> "$LOG" || echo "RESULT: FAIL (script) $c" >> "$LOG"
done
cat "$LOG"
