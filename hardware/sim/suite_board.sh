#!/usr/bin/env bash
# Board-level regression (xsim): driver session replayed through AXI4-Lite on pt_board_top, two clocks.
# usage: bash hardware/sim/suite_board.sh [log]
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG=${1:-$HERE/work/suite_board.log}
mkdir -p "$HERE/work"; : > "$LOG"
#        d NS nsw shots K B  p    seed gmin th32
for c in "3 4 8 8 2 16 0.12 5 0 29" \
         "5 8 64 8 2 16 0.12 5 4 24" \
         "7 16 32 8 2 16 0.15 3 8 29" \
         "3 64 16 8 1 16 0.12 5 4 26"; do
  set -- $c
  bash "$HERE/run_pt_board.sh" "$@" 2>&1 | grep -E "^RESULT" >> "$LOG" || echo "RESULT: FAIL (script) $c" >> "$LOG"
done
cat "$LOG"
