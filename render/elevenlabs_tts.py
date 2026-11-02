#!/usr/bin/env python3
"""Generate audio/scene_N.mp3 from render/scenes.json via the ElevenLabs REST API.

Fallback for when the voiceover was not produced through the ElevenLabs MCP.
Needs ELEVENLABS_API_KEY. Existing files are kept unless --force is given.

  ELEVENLABS_API_KEY=... python3 render/elevenlabs_tts.py [--force] [--model eleven_multilingual_v2]
"""
import argparse, json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", help="override model_id from scenes.json (e.g. eleven_multilingual_v2)")
    a = ap.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("ELEVENLABS_API_KEY is not set")
    cfg = json.load(open(os.path.join(ROOT, "render", "scenes.json")))
    voice = cfg["voice"]["voice_id"]
    model = a.model or cfg["voice"]["model_id"]
    os.makedirs(os.path.join(ROOT, "audio"), exist_ok=True)
    for sc in cfg["scenes"]:
        out = os.path.join(ROOT, sc["audio"])
        if os.path.exists(out) and not a.force:
            print(f"skip  {sc['audio']} (exists)")
            continue
        body = json.dumps({"text": sc["narration"], "model_id": model,
                           "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35,
                                              "use_speaker_boost": True}}).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_192",
            data=body, headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
        with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
            f.write(r.read())
        print(f"wrote {sc['audio']}")

if __name__ == "__main__":
    main()
