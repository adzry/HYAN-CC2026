#!/usr/bin/env python3
"""Generate the per-scene voiceover MP3s from a scenes config via the ElevenLabs REST API.

Needs ELEVENLABS_API_KEY. Existing files are kept unless --force is given.

  ELEVENLABS_API_KEY=... python3 render/elevenlabs_tts.py [--config render/scenes_ms.json] [--force]
"""
import argparse, json, os, sys, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SETTINGS = {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "render", "scenes.json"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", help="override the config's model_id")
    a = ap.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("ELEVENLABS_API_KEY is not set")
    cfg = json.load(open(a.config))
    voice = cfg["voice"]["voice_id"]
    model = a.model or cfg["voice"]["model_id"]
    settings = cfg["voice"].get("voice_settings", DEFAULT_SETTINGS)
    lang = cfg.get("lang")
    for sc in cfg["scenes"]:
        out = os.path.join(ROOT, sc["audio"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and not a.force:
            print(f"skip  {sc['audio']} (exists)")
            continue
        payload = {"text": sc["narration"], "model_id": model, "voice_settings": settings}
        if lang and model in ("eleven_turbo_v2_5", "eleven_flash_v2_5"):
            payload["language_code"] = lang
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
            data=json.dumps(payload).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
                f.write(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"ElevenLabs error {e.code} for {sc['audio']}: {e.read().decode(errors='replace')[:300]}")
        print(f"wrote {sc['audio']}")

if __name__ == "__main__":
    main()
