#!/bin/zsh
# V2b tier renders (AB-T-0376), one process at a time, solo: bf16 / lm8 / lm4 on draws A (seeds 1001–1012) and B (2001–2012),
# c16 = the bf16 takes' codes re-decoded with the fp16 codec (same noise), cfg3 = bf16 at CFG 3.0 on draw A (reported).
# A 10 s overlap log records any other model process on the GPU (`_logs/v2b_overlap.log`). Resumable (manifest rows skip).
#   run_tiers.sh <area-root>   (MLX_ENABLE_TF32=0 is set here for every process — AB-L-0315)
set -u
A=${1:-/Volumes/Satechi/Development/mlxengine-audio}
G=$A/PROD/HeartMuLa/mlx-heartmula-swift/.build/out/Products/Release/heartmula-gates
W=$A/WIP/heartmula-port/weights; O=$A/WIP/heartmula-eval/_out; L=$A/WIP/heartmula-eval/_logs; C=$A/WIP/acestep-eval/cues
mkdir -p $L
export MLX_ENABLE_TF32=0
( while true; do
    # on the full arguments (macOS truncates `comm` when `args` follows it — AB-L-0317; the V2b run used `comm` and could not
    # see a Gaea worker): any *-gates binary but ours, any *-worker, any python whose arguments look like a model job
    o=$(ps -Ao pid=,args= | awk '$0 ~ /-worker( |$)/ || ($0 ~ /-gates( |$)/ && $0 !~ /heartmula-gates/) || ($0 ~ /[Pp]ython/ && $0 ~ /(torch|mlx|harness|eval|run_|dump|score|generate)/)' | grep -v awk | cut -c1-200)
    [ -n "$o" ] && echo "$(date +%H:%M:%S) OVERLAP $o" >> $L/v2b_overlap.log
    sleep 10
  done ) &
WATCH=$!
trap "kill $WATCH 2>/dev/null" EXIT
echo "$(date +%H:%M:%S) start" >> $L/v2b_overlap.log
run() { local tier=$1 draw=$2 lm=$3 codec=$4 prompts=$5; shift 5
  echo "$(date +%H:%M:%S) begin $tier-$draw" >> $L/v2b_overlap.log
  $G batch $W/$lm $W/$codec $C/$prompts $O/heartmula-swift-$tier-$draw --tier $tier --max 120 "$@" >> $L/batch-$tier-$draw.log 2>&1
  echo "$(date +%H:%M:%S) end $tier-$draw rc=$?" >> $L/v2b_overlap.log; }
run bf16 a HeartMuLa-oss-3B-bf16 HeartCodec-oss-fp32 prompts.json
run bf16 b HeartMuLa-oss-3B-bf16 HeartCodec-oss-fp32 prompts-b.json
run lm8 a HeartMuLa-oss-3B-8bit HeartCodec-oss-fp32 prompts.json
run lm8 b HeartMuLa-oss-3B-8bit HeartCodec-oss-fp32 prompts-b.json
run lm4 a HeartMuLa-oss-3B-4bit HeartCodec-oss-fp32 prompts.json
run lm4 b HeartMuLa-oss-3B-4bit HeartCodec-oss-fp32 prompts-b.json
for d in a b; do
  echo "$(date +%H:%M:%S) begin c16-$d" >> $L/v2b_overlap.log
  $G decode $W/HeartMuLa-oss-3B-bf16 $W/HeartCodec-oss-fp16 $O/heartmula-swift-bf16-$d $O/heartmula-swift-c16-$d --tier c16 >> $L/decode-c16-$d.log 2>&1
  echo "$(date +%H:%M:%S) end c16-$d rc=$?" >> $L/v2b_overlap.log
done
run cfg3 a HeartMuLa-oss-3B-bf16 HeartCodec-oss-fp32 prompts.json --cfg 3.0
echo "$(date +%H:%M:%S) DONE" >> $L/v2b_overlap.log
