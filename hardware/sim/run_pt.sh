#!/usr/bin/env bash
# hardware/sim/run_pt.sh <d> [NS] [nsw] [shots] [K] [B] [p] [seed] [gmin] [th32]  -> PASS/FAIL
#   work dir hardware/sim/work/pt_d<d>_ns<NS>_k<K>_b<B>_n<nsw>_s<seed>_g<gmin>; gmin > 0 enables early stopping (threshold th32 / 32)
set -e
D=${1:-3}; NS=${2:-4}; NSW=${3:-8}; SH=${4:-3}; K=${5:-1}; B=${6:-16}; P=${7:-0.12}; SEED=${8:-5}; GMIN=${9:-0}; TH=${10:-29}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
# Vivado simulator tools from $VIVADO_BIN if set, else from PATH (xvlog.bat etc. on Windows); Python from $PYTHON or PATH
tool() { local t="${VIVADO_BIN:+$VIVADO_BIN/}$1"; shift; if command -v "$t.bat" > /dev/null 2>&1; then "$t.bat" "$@"; else "$t" "$@"; fi; }
PY=${PYTHON:-python}
W="$HERE/work/pt_d${D}_ns${NS}_k${K}_b${B}_n${NSW}_s${SEED}_g${GMIN}"
mkdir -p "$W"
"$PY" "$HERE/gen_pt_case.py" --d "$D" --NS "$NS" --nsw "$NSW" --shots "$SH" --K "$K" --B "$B" --p "$P" --seed "$SEED" \
    --gmin "$GMIN" --th32 "$TH" --out "$W" > "$W/gen.log"
cd "$W"
tool xvlog -sv "$ROOT/hardware/rtl/tp_popcnt.sv" "$ROOT/hardware/rtl/tp_lut_rom.sv" "$ROOT/hardware/rtl/pt_xoshiro.sv" "$ROOT/hardware/rtl/pt_thr.sv" "$ROOT/hardware/rtl/pt_units.sv" pt_core.sv "$ROOT/hardware/rtl/pt_top.sv" \
    "$HERE/tb_pt.sv" -i . > xvlog.log 2>&1 || { grep -E "ERROR" xvlog.log | head -20; echo "FAIL (compile)"; exit 1; }
tool xelab tb_pt -s sim -debug off -timescale 1ns/1ps > xelab.log 2>&1 || { grep -E "ERROR" xelab.log | head -20; echo "FAIL (elaborate)"; exit 1; }
tool xsim sim -runall > xsim.log 2>&1 || true
grep -E "^(PASS|FAIL|shot)" xsim.log | head -30
grep -qE "^PASS" xsim.log && echo "RESULT: PASS (pt d=$D NS=$NS nsw=$NSW shots=$SH K=$K B=$B gmin=$GMIN th=$TH)" || { echo "RESULT: FAIL"; exit 1; }
