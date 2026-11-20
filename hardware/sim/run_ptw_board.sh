#!/usr/bin/env bash
# hardware/sim/run_ptw_board.sh <d> [R] [k] [lram 0|1] [rs] [cs 0|1] [nsw] [shots] [K] [p] [seed] [srl 0|1]  -> PASS/FAIL
#   real-time board-level xsim: the PYNQ driver's recorded session (gen_ptw_board_case.py) replayed through AXI4-Lite on
#   ptw_board_top (100 MHz AXI clock, unrelated decoder clock)
set -e
D=${1:-3}; R=${2:-4}; KS=${3:-1}; LR=${4:-0}; RS=${5:-1}; CS=${6:-0}; NSW=${7:-8}; SH=${8:-4}; K=${9:-1}; P=${10:-0.04}
SEED=${11:-5}; SRL=${12:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
# Vivado simulator tools from $VIVADO_BIN if set, else from PATH (xvlog.bat etc. on Windows); Python from $PYTHON or PATH
tool() { local t="${VIVADO_BIN:+$VIVADO_BIN/}$1"; shift; if command -v "$t.bat" > /dev/null 2>&1; then "$t.bat" "$@"; else "$t" "$@"; fi; }
PY=${PYTHON:-python}
W="$HERE/work/pbrd_d${D}_r${R}_k${KS}l${LR}_rs${RS}_x${CS}_K${K}_n${NSW}_s${SEED}"
mkdir -p "$W"
"$PY" "$HERE/gen_ptw_board_case.py" --d "$D" --R "$R" --k "$KS" --lram "$LR" --rs "$RS" --cs "$CS" --srl "$SRL" --K "$K" \
    --nsw "$NSW" --shots "$SH" --p "$P" --seed "$SEED" --out "$W" > "$W/gen.log" 2>&1 || { tail -5 "$W/gen.log"; echo "RESULT: FAIL (vectors)"; exit 1; }
cd "$W"
tool xvlog -sv "$ROOT/hardware/rtl/tp_popcnt.sv" "$ROOT/hardware/rtl/tp_lut_rom.sv" "$ROOT/hardware/rtl/pt_xoshiro.sv" "$ROOT/hardware/rtl/pt_thr.sv" \
    "$ROOT/hardware/rtl/pt_units.sv" "$ROOT/hardware/rtl/ptw_units.sv" ptw_core.sv "$ROOT/hardware/rtl/ptw_top.sv" "$ROOT/hardware/rtl/ptw_board.sv" \
    "$HERE/tb_ptw_board.sv" -i . > xvlog.log 2>&1 || { grep -E "ERROR" xvlog.log | head -20; echo "RESULT: FAIL (compile)"; exit 1; }
tool xvlog "$ROOT/hardware/rtl/ptw_board_top.v" >> xvlog.log 2>&1 || { grep -E "ERROR" xvlog.log | head; echo "RESULT: FAIL (compile)"; exit 1; }
tool xelab tb_ptw_board -s simb -debug off -timescale 1ns/1ps > xelab.log 2>&1 || { grep -E "ERROR" xelab.log | head -20; echo "RESULT: FAIL (elaborate)"; exit 1; }
tool xsim simb -runall > xsim.log 2>&1 || true
grep -E "^(PASS|FAIL|PROTOCOL|INFO)" xsim.log | head -40
TAG="ptw board d=$D R=$R k=$KS lram=$LR rs=$RS cs=$CS K=$K nsw=$NSW shots=$SH p=$P seed=$SEED"
grep -qE "^PASS" xsim.log && echo "RESULT: PASS ($TAG)" || { echo "RESULT: FAIL ($TAG)"; exit 1; }
