#!/bin/zsh
# V3 engine-package lanes (AB-T-0376, WIP/heartmula-port/MEASUREMENTS.md §V3), one process at a time, solo: `validate` (through
# MLXServeEngine) then `envelope` (240 s, end token ignored) for each offered pairing — LM int8 / bf16 × codec fp16 / fp32.
# Before each lane it waits (≤ 20 min) until no other model process is running; a 5 s watcher logs any overlap during a lane
# (`receipts/v3_overlap.log`) so an overlapped lane can be marked and re-timed. phys_footprint is per process either way.
#   run_v3.sh <area-root> [lane-regex] [suffix]   (MLX_ENABLE_TF32=0 for every process — AB-L-0315); e.g. a solo re-time of
#   three lanes: run_v3.sh <root> 'validate-(int8-fp16|int8-fp32|bf16-fp16)' -retime
set -u
A=${1:-/Volumes/Satechi/Development/mlxengine-audio}
G=$A/PROD/HeartMuLa/mlx-heartmula-swift/.build/out/Products/Release/heartmula-gates
ONLY=${2:-.}; SUF=${3:-}
W=$A/WIP/heartmula-port/weights; R=$A/WIP/heartmula-port/receipts; C=$A/WIP/acestep-eval/cues
mkdir -p $R/validate
export MLX_ENABLE_TF32=0
others() {   # any *-gates binary but ours, any *-worker (Gaea plugins), any python that looks like a model job — matched on the
             # full arguments: with `args` after it, macOS truncates `comm` (a Gaea worker reads "/tmp/g371r/main/"), which
             # blinded the first V3 run's watcher
  ps -Ao pid=,args= | awk '$0 ~ /-worker( |$)/ || ($0 ~ /-gates( |$)/ && $0 !~ /heartmula-gates/) || ($0 ~ /[Pp]ython/ && $0 ~ /(torch|mlx|harness|eval|run_|dump|score|generate)/)' | grep -v awk | cut -c1-160
}
( while true; do
    o=$(others); [ -n "$o" ] && echo "$(date +%H:%M:%S) OVERLAP $o" >> $R/v3_overlap.log
    sleep 5
  done ) &
WATCH=$!
trap "kill $WATCH 2>/dev/null" EXIT
waitfree() { local n=0; while [ -n "$(others)" ] && [ $n -lt 240 ]; do sleep 5; n=$((n+1)); done
  [ $n -gt 0 ] && echo "$(date +%H:%M:%S) waited $((n*5)) s for the GPU$( [ -n "$(others)" ] && echo ' — STILL BUSY, running anyway')" >> $R/v3_overlap.log; }
lane() { local name=$1$SUF; shift
  [[ $name =~ $ONLY ]] || return 0
  waitfree
  echo "$(date +%H:%M:%S) begin $name" >> $R/v3_overlap.log
  "$@" > $R/$name.log 2>&1
  echo "$(date +%H:%M:%S) end $name rc=$?" >> $R/v3_overlap.log; }
echo "$(date +%H:%M:%S) start" >> $R/v3_overlap.log
for pair in "8bit fp16 int8-fp16" "8bit fp32 int8-fp32" "bf16 fp16 bf16-fp16" "bf16 fp32 bf16-fp32"; do
  set -- ${=pair}
  lane validate-$3 $G validate $W/HeartMuLa-oss-3B-$1 $W/HeartCodec-oss-$2 $C/prompts.json p04 $R/validate --seed 1004 --max 120
done
for pair in "8bit fp16 int8-fp16" "8bit fp32 int8-fp32" "bf16 fp16 bf16-fp16" "bf16 fp32 bf16-fp32"; do
  set -- ${=pair}
  lane envelope-$3 $G envelope $W/HeartMuLa-oss-3B-$1 $W/HeartCodec-oss-$2 $C/prompts.json
done
echo "$(date +%H:%M:%S) DONE" >> $R/v3_overlap.log
