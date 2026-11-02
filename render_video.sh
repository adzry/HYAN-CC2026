#!/usr/bin/env bash
# Render the Hospital Yan Community Carnival 2026 Vox-style video.
#
#   ./render_video.sh                 # full 1920x1080 @ 30fps -> out/hyan_carnival_2026.mp4
#   ./render_video.sh --preview       # quick 960x540 @ 15fps check
#   ./render_video.sh --music bed.mp3 # add a music bed, auto-ducked under the voice
#   ./render_video.sh --no-captions   # burn-in captions off
#   ./render_video.sh --still 42      # write a single PNG frame at 42s
#
# Voiceover: audio/scene_1.mp3 ... scene_5.mp3. Missing scenes render silent at
# their target length. If ELEVENLABS_API_KEY is set, missing scenes are
# generated first via render/elevenlabs_tts.py.
set -euo pipefail
cd "$(dirname "$0")"

for bin in python3 ffmpeg ffprobe; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done
python3 -c "import PIL, numpy" 2>/dev/null || {
  echo "installing Python deps (Pillow, numpy)…" >&2
  python3 -m pip install --quiet -r render/requirements.txt
}

missing=0
for n in 1 2 3 4 5; do [[ -f "audio/scene_$n.mp3" ]] || missing=1; done
if [[ $missing == 1 && -n "${ELEVENLABS_API_KEY:-}" ]]; then
  python3 render/elevenlabs_tts.py
fi

exec python3 render/render.py "$@"
