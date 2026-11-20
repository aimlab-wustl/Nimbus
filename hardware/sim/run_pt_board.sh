#!/usr/bin/env bash
# hardware/sim/run_pt_board.sh <d> [NS] [nsw] [shots, multiple of 8] [K] [B] [p] [seed] [gmin] [th32]  -> PASS/FAIL
#   board-level xsim: the PYNQ driver's recorded session (gen_board_case.py) replayed through AXI4-Lite on pt_board_top
set -e
D=${1:-3}; NS=${2:-4}; NSW=${3:-8}; SH=${4:-8}; K=${5:-2}; B=${6:-16}; P=${7:-0.12}; SEED=${8:-5}; GMIN=${9:-0}; TH=${10:-29}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
# Vivado simulator tools from $VIVADO_BIN if set, else from PATH (xvlog.bat etc. on Windows); Python from $PYTHON or PATH
tool() { local t="${VIVADO_BIN:+$VIVADO_BIN/}$1"; shift; if command -v "$t.bat" > /dev/null 2>&1; then "$t.bat" "$@"; else "$t" "$@"; fi; }
PY=${PYTHON:-python}
W="$HERE/work/brd_d${D}_ns${NS}_k${K}_b${B}_n${NSW}_s${SEED}_g${GMIN}"
mkdir -p "$W"
"$PY" "$HERE/gen_board_case.py" --d "$D" --NS "$NS" --nsw "$NSW" --shots "$SH" --K "$K" --B "$B" --p "$P" --seed "$SEED" \
    --gmin "$GMIN" --th32 "$TH" --out "$W" > "$W/gen.log" || { tail -5 "$W/gen.log"; echo "FAIL (vectors)"; exit 1; }
cd "$W"
tool xvlog -sv "$ROOT/hardware/rtl/tp_popcnt.sv" "$ROOT/hardware/rtl/tp_lut_rom.sv" "$ROOT/hardware/rtl/pt_xoshiro.sv" "$ROOT/hardware/rtl/pt_thr.sv" "$ROOT/hardware/rtl/pt_units.sv" \
    pt_core.sv "$ROOT/hardware/rtl/pt_top.sv" "$ROOT/hardware/rtl/pt_board.sv" "$HERE/tb_pt_board.sv" -i . > xvlog.log 2>&1 \
    || { grep -E "ERROR" xvlog.log | head -20; echo "FAIL (compile)"; exit 1; }
tool xvlog "$ROOT/hardware/rtl/pt_board_top.v" >> xvlog.log 2>&1 || { grep -E "ERROR" xvlog.log | head; echo "FAIL (compile)"; exit 1; }
tool xelab tb_pt_board -s simb -debug off -timescale 1ns/1ps > xelab.log 2>&1 || { grep -E "ERROR" xelab.log | head -20; echo "FAIL (elaborate)"; exit 1; }
tool xsim simb -runall > xsim.log 2>&1 || true
grep -E "^(PASS|FAIL|PROTOCOL)" xsim.log | head -30
grep -qE "^PASS" xsim.log && echo "RESULT: PASS (board d=$D NS=$NS nsw=$NSW shots=$SH K=$K B=$B gmin=$GMIN th=$TH)" || { echo "RESULT: FAIL"; exit 1; }
