#!/usr/bin/env bash
# Render the Hospital Yan Community Carnival 2026 Vox-style video.
#
#   ./render_video.sh                  # English  -> out/hyan_carnival_2026.mp4    (1920x1080 @ 30fps)
#   ./render_video.sh --lang ms        # Melayu   -> out/hyan_carnival_2026_ms.mp4 (with music bed)
#   ./render_video.sh --lang ms --preview      # quick 960x540 check
#   ./render_video.sh --lang ms --still 42     # one PNG frame at 42s
#   ./render_video.sh --whatsapp out/hyan_carnival_2026_ms.mp4   # ~70 MB 1080p copy for WhatsApp
#
# Voiceover lives in audio/ (English) and audio/ms/ (Melayu). Missing scenes
# render silent; if ELEVENLABS_API_KEY is set they are generated first.
set -euo pipefail
cd "$(dirname "$0")"

for bin in python3 ffmpeg ffprobe; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done
python3 -c "import PIL, numpy" 2>/dev/null || {
  echo "installing Python deps (Pillow, numpy)…" >&2
  python3 -m pip install --quiet -r render/requirements.txt
}

if [[ "${1:-}" == "--whatsapp" ]]; then
  in="${2:?usage: --whatsapp input.mp4}"
  out="${in%.mp4}_whatsapp.mp4"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
  # ~72 MB total: video bitrate from the duration, 128k AAC
  vb=$(python3 -c "print(int((72*8*1024)/$dur - 128))")
  log=$(mktemp -d)/x264
  vf="hqdn3d=1.5:1.5:4:4"
  ffmpeg -v error -y -i "$in" -vf "$vf" -c:v libx264 -preset slow -profile:v high -level 4.1 -pix_fmt yuv420p \
    -b:v "${vb}k" -pass 1 -passlogfile "$log" -an -f mp4 /dev/null
  ffmpeg -v error -y -i "$in" -vf "$vf" -c:v libx264 -preset slow -profile:v high -level 4.1 -pix_fmt yuv420p \
    -b:v "${vb}k" -maxrate "$((vb * 3 / 2))k" -bufsize "$((vb * 2))k" -pass 2 -passlogfile "$log" \
    -c:a aac -b:a 128k -movflags +faststart "$out"
  echo "[done] $out ($(du -h "$out" | cut -f1))"
  exit 0
fi

lang=en
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[$i]}" == "--lang" ]] && lang="${args[$((i + 1))]}"
done
cfg=render/scenes.json; dir=audio
[[ "$lang" != en ]] && { cfg="render/scenes_${lang}.json"; dir="audio/${lang}"; }

missing=0
for n in 1 2 3 4 5; do [[ -f "$dir/scene_$n.mp3" ]] || missing=1; done
if [[ $missing == 1 && -n "${ELEVENLABS_API_KEY:-}" ]]; then
  python3 render/elevenlabs_tts.py --config "$cfg"
fi

exec python3 render/render.py "$@"
