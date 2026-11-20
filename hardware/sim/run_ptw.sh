#!/usr/bin/env bash
# hardware/sim/run_ptw.sh <d> [R] [k] [lram 0|1] [nsw] [shots] [K] [p] [seed] [rs] [srl 0|1] [carry 0|1] [cs 0|1] [swp 0|1]
#   -> PASS/FAIL (real-time core vs nimbus.decoding.real_time); work dir hardware/sim/work/ptw_d<d>_r<R>_k<k>[l]_rs<rs>_K<K>_n<nsw>_s<seed>...
set -e
D=${1:-3}; R=${2:-4}; KS=${3:-1}; LR=${4:-0}; NSW=${5:-8}; SH=${6:-2}; K=${7:-1}; P=${8:-0.04}; SEED=${9:-5}
RS=${10:-1}; SRL=${11:-0}; CARRY=${12:-1}; CS=${13:-0}; SWP=${14:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
# Vivado simulator tools from $VIVADO_BIN if set, else from PATH (xvlog.bat etc. on Windows); Python from $PYTHON or PATH
tool() { local t="${VIVADO_BIN:+$VIVADO_BIN/}$1"; shift; if command -v "$t.bat" > /dev/null 2>&1; then "$t.bat" "$@"; else "$t" "$@"; fi; }
PY=${PYTHON:-python}
LFLAG=""; LT=""; if [ "$LR" = "1" ]; then LFLAG="--lram"; LT="l"; fi
W="$HERE/work/ptw_d${D}_r${R}_k${KS}${LT}_rs${RS}_K${K}_n${NSW}_s${SEED}_c${CARRY}_q${SRL}_x${CS}_w${SWP}"
mkdir -p "$W"
"$PY" "$HERE/gen_ptw_case.py" --d "$D" --R "$R" --k "$KS" $LFLAG --K "$K" --nsw "$NSW" --shots "$SH" --p "$P" --seed "$SEED" \
    --rs "$RS" --srl "$SRL" --carry "$CARRY" --cs "$CS" --swp "$SWP" --out "$W" > "$W/gen.log" 2>&1 || { tail -5 "$W/gen.log"; echo "RESULT: FAIL (case)"; exit 1; }
cd "$W"
tool xvlog -sv "$ROOT/hardware/rtl/tp_popcnt.sv" "$ROOT/hardware/rtl/tp_lut_rom.sv" "$ROOT/hardware/rtl/pt_xoshiro.sv" "$ROOT/hardware/rtl/pt_thr.sv" \
    "$ROOT/hardware/rtl/pt_units.sv" "$ROOT/hardware/rtl/ptw_units.sv" ptw_core.sv "$ROOT/hardware/rtl/ptw_top.sv" "$HERE/tb_ptw.sv" -i . \
    > xvlog.log 2>&1 || { grep -E "ERROR" xvlog.log | head -20; echo "RESULT: FAIL (compile)"; exit 1; }
tool xelab tb_ptw -s sim -debug off -timescale 1ns/1ps > xelab.log 2>&1 || { grep -E "ERROR" xelab.log | head -20; echo "RESULT: FAIL (elaborate)"; exit 1; }
tool xsim sim -runall > xsim.log 2>&1 || true
grep -E "^(PASS|FAIL|shot)" xsim.log | head -40
TAG="ptw d=$D R=$R k=$KS lram=$LR rs=$RS srl=$SRL carry=$CARRY cs=$CS swp=$SWP K=$K nsw=$NSW shots=$SH p=$P seed=$SEED"
grep -qE "^PASS" xsim.log && echo "RESULT: PASS ($TAG)" || { echo "RESULT: FAIL ($TAG)"; exit 1; }
