# Hospital Yan Community Carnival 2026 — Vox-style video pipeline

2:45 explainer video for the Hospital Director, built from the booth list
*SENARAI BOOTH PAMERAN – HOSPITAL YAN COMMUNITY CARNIVAL 2026* (18 Nov 2026).

| Path | What it is |
|---|---|
| `video_manifest.json` | 12 booths indexed (games, patient-safety initiatives, collaborations), every image matched to a booth, and the image list for each scene |
| `storyboard.md` | 401-word, 5-scene voiceover script with a shot-by-shot storyboard |
| `images/` | 29 images extracted from the source .docx |
| `audio/scene_1..5.mp3` | ElevenLabs voiceover (voice: *Connery – Deep & Immersive Documentary*) |
| `render/scenes.json` | Narration text, voice and target timings (single source for TTS and renderer) |
| `render/render.py` | Pillow + FFmpeg renderer: dark theme, kinetic type, Ken Burns photo cards, chapter cards, progress bar, captions |
| `render/elevenlabs_tts.py` | REST fallback that generates the MP3s with `ELEVENLABS_API_KEY` |
| `render_video.sh` | One-command render |

## Render

```bash
./render_video.sh            # -> out/hyan_carnival_2026.mp4 (1920x1080, 30 fps)
./render_video.sh --preview  # fast 960x540 check
./render_video.sh --music bed.mp3   # optional music, auto-ducked under the voice
```

Needs `python3` (Pillow, numpy), `ffmpeg` and `ffprobe`. Each scene runs as long
as its MP3, and visual cues are placed where the narrator says the matching
phrase. A missing MP3 renders silent at the scene's target length.

## Open items
- No photos yet for Booth 7 (CPR/AED) or the Booth 4 hand-hygiene activity; scene 3 uses kinetic type there.
- `images/image22.jpg` (AADK logo) is in the doc but AADK is not a listed exhibitor, so it is unused.
- Images come from the 1 Nov copy of the doc. The 2 Nov original (13 MB) was too large for the Drive connector; any newer photos in it are not included.
