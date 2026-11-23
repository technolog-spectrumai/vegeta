#!/usr/bin/env bash
# Render the fire scene (the monkey, Chuck, the ladder climb, the fire, the radio call) to
# output/space_spider_fire.mp4, in parallel chunks.
#
#   ./ladder.sh            # 4 processes
#   JOBS=8 ./ladder.sh     # more processes on a bigger machine
#   PASSES=8 ./ladder.sh   # more solidity passes (slower, tighter contacts; default 2)
#
# Needs python with mujoco and the repo's requirements, and ffmpeg. Each process builds
# the scene first (~25 min), then renders its share of the ~1070 frames (~13 s each).
set -euo pipefail
cd "$(dirname "$0")"

JOBS=${JOBS:-4}
PASSES=${PASSES:-2}
WORK=${WORK:-$(mktemp -d -t fire_XXXX)}
FRAMES="$WORK/frames"
OUT=output/space_spider_fire.mp4
mkdir -p "$FRAMES" output

echo "rendering with $JOBS processes; frames and logs in $WORK"
pids=()
for ((i = 0; i < JOBS; i++)); do
  VEGETA_SOLID_PASSES=$PASSES python -u space_kerchak_scenes.py --scene fire --machine spider \
    --render mesh --casing plastic --chunk "$i/$JOBS" --frames-dir "$FRAMES" > "$WORK/chunk$i.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "chunk $i failed, see $WORK/chunk$i.log"; fail=1
  fi
done
[ "$fail" = 0 ] || exit 1

echo "$(ls "$FRAMES" | wc -l) frames; assembling $OUT"
ffmpeg -y -loglevel error -framerate 10 -i "$FRAMES/f%04d.png" -c:v libx264 -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$OUT"
echo "done: $(pwd)/$OUT"
