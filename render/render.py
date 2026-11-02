#!/usr/bin/env python3
"""Vox-style renderer (v2) for the Hospital Yan Community Carnival 2026 video.

Draws every frame with Pillow (dark grid theme, mask-reveal kinetic type,
Ken Burns photo cards, callout lines, chapter cards, skewed wipe transitions,
progress bar, captions) and pipes them into FFmpeg. Audio is mixed in numpy:
narration, a music bed that ducks under the voice and swells in pauses and
chapter transitions, and synthesized whoosh/impact/typewriter effects.

Each language is a config in render/: scenes.json (English, default) and
scenes_ms.json (Bahasa Melayu). A config holds the narration, the cue phrases
that time each visual, the on-screen strings and the chapter cards.

Timing: if audio/<...>/scene_N.words.json (ElevenLabs Scribe word timestamps)
exists, cues and captions follow the spoken words; otherwise the narrator's
pauses (FFmpeg silencedetect) are aligned to the script's punctuation.

Usage (from the repo root; normally via ./render_video.sh):
  python3 render/render.py [--lang en|ms] [--out out/video.mp4] [--preview]
                           [--still SECONDS ...] [--music bed.mp3 | --no-music]
                           [--no-captions] [--jobs N]
"""
import argparse, difflib, json, math, os, re, shutil, subprocess, sys, tempfile
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "render")
W, H = 1920, 1080
SR = 48000
WIPE = 0.42         # half-length of the chapter wipe transition (s)

BG = (14, 15, 18)
INK = (240, 237, 228)
DIM = (150, 150, 158)
YEL = (255, 210, 63)
RED = (232, 72, 60)
TEAL = (64, 196, 180)
NAVY = (24, 28, 44)
CAT_COL = {"Hospital Yan": YEL, "JKN": RED, "Agensi Luar": TEAL, "Komuniti": (150, 160, 255)}


# --------------------------------------------------------------------- helpers
def font(name, size):
    return _font(name, int(size))

@lru_cache(maxsize=None)
def _font(name, size):
    files = {"black": "InterDisplay-Black.otf", "bold": "InterDisplay-Bold.otf",
             "semi": "Inter-SemiBold.otf", "med": "Inter-Medium.otf", "mono": "JetBrainsMono-Bold.ttf"}
    return ImageFont.truetype(os.path.join(HERE, "fonts", files[name]), size)

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x

def ease_out(x):
    x = clamp(x); return 1 - (1 - x) ** 3

def ease_io(x):
    x = clamp(x); return 3 * x * x - 2 * x * x * x

def ease_expo(x):
    x = clamp(x); return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)

def ease_back(x, s=1.6):
    x = clamp(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2

def env(t, a, b, fi=0.35, fo=0.3):
    """Opacity envelope for an element visible from a to b (seconds)."""
    if t < a or t > b:
        return 0.0
    return min(clamp((t - a) / fi) if fi else 1.0, clamp((b - t) / fo) if fo else 1.0)

@lru_cache(maxsize=4096)
def text_img(text, fname, size, color=INK, tracking=0):
    f = font(fname, size)
    if tracking:
        widths = [f.getlength(c) + tracking for c in text]
        w = int(sum(widths)) + 4
    else:
        w = int(f.getlength(text)) + 4
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (max(w, 1), asc + desc + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if tracking:
        x = 0
        for c, cw in zip(text, widths):
            d.text((x, 0), c, font=f, fill=color); x += cw
    else:
        d.text((0, 0), text, font=f, fill=color)
    return im

def fit_size(text, fname, size, max_w):
    w = text_img(text, fname, size).width
    return size if w <= max_w else max(12, int(size * max_w / w))

def paste(canvas, im, xy, alpha=1.0):
    if alpha <= 0.003:
        return
    x, y = int(round(xy[0])), int(round(xy[1]))
    if alpha < 0.997:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    l, t = max(0, -x), max(0, -y)
    r, b = min(im.width, W - x), min(im.height, H - y)
    if r <= l or b <= t:
        return
    if (l, t, r, b) != (0, 0, im.width, im.height):
        im = im.crop((l, t, r, b))
    canvas.alpha_composite(im, (x + l, y + t))

def rect(canvas, box, color, alpha=1.0):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    paste(canvas, Image.new("RGBA", (x1 - x0, y1 - y0), color + (255,)), (x0, y0), alpha)

def reveal(canvas, im, x, y, p, alpha=1.0):
    """Mask reveal: the image rises out of an invisible baseline at y + height."""
    p = clamp(p)
    h = int(im.height * p)
    if h <= 0:
        return
    paste(canvas, im.crop((0, 0, im.width, h)), (x, y + im.height - h), alpha)

def label(canvas, text, xy, t_in, t, size=34, bg=YEL, fg=(12, 12, 14), alpha=1.0):
    """Yellow tag that wipes in from the left, text revealed behind the wipe."""
    p = ease_expo((t - t_in) / 0.5)
    if p <= 0:
        return
    ti = text_img(text, "black", size, fg)
    pad = int(size * 0.45)
    w, h = ti.width + pad * 2, ti.height + pad
    rect(canvas, (xy[0], xy[1], xy[0] + w * p, xy[1] + h), bg, alpha)
    q = clamp((p - 0.3) / 0.7)
    if q > 0:
        reveal(canvas, ti.crop((0, 0, max(1, int(ti.width * q)), ti.height)), xy[0] + pad, xy[1] + pad * 0.35, ease_out(q * 1.4), alpha)

def subline(canvas, text, xy, t_in, t, size=30, color=INK, alpha=1.0):
    p = ease_out((t - t_in) / 0.55)
    paste(canvas, text_img(text, "semi", size, color), (xy[0] - 36 * (1 - p), xy[1]), alpha * p)

def kinetic(canvas, text, cx, cy, t_in, t, size=110, color=INK, fname="black", alpha=1.0, stagger=0.06,
            max_w=1700, draw=True):
    """Word-by-word mask-reveal headline centred on (cx, cy). Returns word boxes."""
    size = fit_size(text, fname, size, max_w)
    words = text.split(" ")
    imgs = [text_img(w, fname, size, color) for w in words]
    space = font(fname, size).getlength(" ")
    total = sum(i.width for i in imgs) + space * (len(imgs) - 1)
    x = cx - total / 2
    boxes = []
    for k, (w, im) in enumerate(zip(words, imgs)):
        p = ease_expo((t - t_in - k * stagger) / 0.55)
        y = cy - im.height / 2
        if draw:
            reveal(canvas, im, x, y, p, alpha)
        boxes.append((w, x, y, im.width, im.height))
        x += im.width + space
    return boxes

def highlighter(canvas, box, t_in, t, color=YEL, alpha=0.9, dur=0.45):
    x, y, w, h = box
    p = ease_expo((t - t_in) / dur)
    if p > 0:
        rect(canvas, (x - 12, y + h * 0.2, x - 12 + (w + 24) * p, y + h * 0.93), color, alpha)

def line(canvas, p0, p1, prog, color=YEL, width=4, alpha=1.0, dash=0):
    prog = clamp(prog)
    if prog <= 0 or alpha <= 0:
        return
    x1 = p0[0] + (p1[0] - p0[0]) * prog
    y1 = p0[1] + (p1[1] - p0[1]) * prog
    pad = width * 2 + 2
    bx0, by0 = int(min(p0[0], x1)) - pad, int(min(p0[1], y1)) - pad
    bx1, by1 = int(max(p0[0], x1)) + pad, int(max(p0[1], y1)) + pad
    layer = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    q0, q1 = (p0[0] - bx0, p0[1] - by0), (x1 - bx0, y1 - by0)
    col = color + (int(255 * clamp(alpha)),)
    if dash:
        L = math.hypot(q1[0] - q0[0], q1[1] - q0[1])
        for k in range(int(L // (dash * 2)) + 1):
            a0, a1 = k * 2 * dash / max(L, 1), min(1, (k * 2 * dash + dash) / max(L, 1))
            d.line([(q0[0] + (q1[0] - q0[0]) * a0, q0[1] + (q1[1] - q0[1]) * a0),
                    (q0[0] + (q1[0] - q0[0]) * a1, q0[1] + (q1[1] - q0[1]) * a1)], fill=col, width=width)
    else:
        d.line([q0, q1], fill=col, width=width)
    paste(canvas, layer, (bx0, by0))

@lru_cache(maxsize=32)
def _dot(r, ring):
    s = int(r * 2 + ring * 2 + 8)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = s / 2
    if ring:
        d.ellipse((c - r - ring, c - r - ring, c + r + ring, c + r + ring), outline=YEL + (255,), width=3)
    d.ellipse((c - r, c - r, c + r, c + r), fill=YEL + (255,), outline=BG + (255,), width=3)
    return im

def callout(canvas, p0, p1, text, t_in, t, alpha=1.0, size=30, sub=None):
    """Modern pointer: pulsing dot on the subject, angled leader line, underline and label."""
    if t < t_in or alpha <= 0:
        return
    pd = ease_back((t - t_in) / 0.35)
    if pd > 0:
        im = _dot(9, 0)
        paste(canvas, im, (p0[0] - im.width / 2, p0[1] - im.height / 2), alpha * clamp(pd))
        pulse = ((t - t_in) % 1.4) / 1.4
        ring = _dot(9, int(4 + 22 * pulse))
        paste(canvas, ring, (p0[0] - ring.width / 2, p0[1] - ring.height / 2), alpha * (1 - pulse) * 0.7)
    line(canvas, p0, p1, ease_io((t - t_in - 0.15) / 0.4), YEL, 3, alpha)
    ti = text_img(text, "black", size, INK)
    dirn = 1 if p1[0] >= p0[0] else -1
    u = ease_io((t - t_in - 0.5) / 0.35)
    if u > 0:
        x_end = p1[0] + dirn * (ti.width + 20) * u
        rect(canvas, (min(p1[0], x_end), p1[1] - 1.5, max(p1[0], x_end), p1[1] + 1.5), YEL, alpha)
    tx = p1[0] + 10 if dirn > 0 else p1[0] - 10 - ti.width
    if u > 0:
        rect(canvas, (tx - 8, p1[1] - ti.height - 6, tx + ti.width + 6, p1[1] - 2), BG, alpha * 0.82 * u)
    reveal(canvas, ti, tx, p1[1] - ti.height - 4, ease_expo((t - t_in - 0.6) / 0.45), alpha)
    if sub:
        si = text_img(sub, "med", int(size * 0.75), DIM)
        sx = p1[0] + 10 if dirn > 0 else p1[0] - 10 - si.width
        paste(canvas, si, (sx, p1[1] + 8), alpha * ease_out((t - t_in - 0.8) / 0.4))


# ------------------------------------------------------------- photo cards
@lru_cache(maxsize=64)
def src_img(name):
    return Image.open(os.path.join(ROOT, "images", name)).convert("RGB")

@lru_cache(maxsize=64)
def shadow(w, h, blur=28):
    pad = blur * 3
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((pad, pad + 14, pad + w, pad + h + 14), fill=(0, 0, 0, 175))
    return im.filter(ImageFilter.GaussianBlur(blur)), pad

def card(canvas, name, cx, cy, max_w, max_h, t_in, t_out, t, border=10, zoom=0.10, drift=(0.5, 0.5),
         slide=(0, 70), alpha=1.0, float_px=14):
    """Photo on a paper card: eased Ken Burns inside, slow float of the card itself."""
    a = env(t, t_in, t_out, 0.4, 0.3) * alpha
    if a <= 0:
        return None
    src = src_img(name)
    sw, sh = src.size
    s = min((max_w - 2 * border) / sw, (max_h - 2 * border) / sh)
    iw, ih = int(sw * s), int(sh * s)
    life = ease_io(clamp((t - t_in) / max(0.1, t_out - t_in)))
    z = 1 + zoom * life
    cw, ch = sw / z, sh / z
    ox = (sw - cw) * (0.5 + (drift[0] - 0.5) * life)
    oy = (sh - ch) * (0.5 + (drift[1] - 0.5) * life)
    photo = src.resize((iw, ih), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))
    w, h = iw + 2 * border, ih + 2 * border
    p = ease_expo((t - t_in) / 0.7)
    q = ease_io((t - (t_out - 0.3)) / 0.3)
    x = cx - w / 2 + slide[0] * (1 - p) - float_px * (life - 0.5)
    y = cy - h / 2 + slide[1] * (1 - p) - 30 * q
    sh_im, pad = shadow(w, h)
    paste(canvas, sh_im, (x - pad, y - pad), a * 0.9)
    paperc = Image.new("RGBA", (w, h), (246, 243, 236, 255))
    paperc.paste(photo, (border, border))
    paste(canvas, paperc, (x, y), a)
    return (x, y, w, h)

def pt(box, fx, fy):
    """Point inside a card box at fractional position (fx, fy)."""
    if not box:
        return (0, 0)
    x, y, w, h = box
    return (x + w * fx, y + h * fy)

def logo_disc(canvas, name, cx, cy, r, t_in, t, alpha=1.0):
    p = ease_back((t - t_in) / 0.5)
    if p <= 0:
        return
    rr = max(2, int(r * (0.5 + 0.5 * p)))
    im = _logo_disc(name, rr)
    paste(canvas, im, (cx - im.width / 2, cy - im.height / 2), alpha * clamp(p))

@lru_cache(maxsize=256)
def _logo_disc(name, r):
    src = src_img(name)
    d = 2 * r
    out = Image.new("RGBA", (d + 12, d + 12), (0, 0, 0, 0))
    ImageDraw.Draw(out).ellipse((0, 0, d + 11, d + 11), fill=YEL + (255,))
    disc = Image.new("RGBA", (d, d), (255, 255, 255, 255))
    s = min(d * 0.74 / src.width, d * 0.74 / src.height)
    lg = src.resize((max(1, int(src.width * s)), max(1, int(src.height * s))), Image.LANCZOS)
    disc.paste(lg, ((d - lg.width) // 2, (d - lg.height) // 2))
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, d - 1, d - 1), fill=255)
    disc.putalpha(mask)
    out.alpha_composite(disc, (6, 6))
    return out


# ------------------------------------------------------------- global layers
@lru_cache(maxsize=1)
def grain_layers():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(6):
        n = rng.integers(0, 255, (H // 2, W // 2), dtype=np.uint8)
        a = np.full_like(n, 12)
        out.append(Image.fromarray(np.dstack([n, n, n, a]), "RGBA").resize((W, H), Image.NEAREST))
    return out

@lru_cache(maxsize=1)
def vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    a = (np.clip((d - 0.65) / 0.75, 0, 1) ** 1.6 * 200).astype(np.uint8)
    z = np.zeros_like(a)
    return Image.fromarray(np.dstack([z, z, z, a]), "RGBA")

@lru_cache(maxsize=1)
def backdrop():
    im = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 120):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 7))
    for y in range(0, H, 120):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 7))
    return im


# ------------------------------------------------------------------ timing
def norm_tok(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())

class Timing:
    """Maps a character index in the narration to seconds into the (tempo-adjusted) voiceover.

    Preferred: Scribe word timestamps aligned to the script with difflib.
    Fallback: narrator pauses (silencedetect) matched to punctuation by DP.
    Without audio: linear.
    """
    def __init__(self, text, dur, silences=None, words=None, tempo=1.0):
        n = len(text)
        if words:
            xs, ys = self._from_words(text, words)
        else:
            xs, ys = self._from_silences(text, dur * tempo, silences)
        order = np.argsort(xs, kind="stable")
        self.xs = np.array(xs, float)[order]
        self.ys = np.array(ys, float)[order] / tempo

    @staticmethod
    def _from_words(text, words):
        toks = [(m.start(), m.end(), norm_tok(m.group())) for m in re.finditer(r"\S+", text)]
        toks = [t for t in toks if t[2]]
        ws = [w for w in words if w.get("type", "word") == "word" and norm_tok(w["text"])]
        a, b = [t[2] for t in toks], [norm_tok(w["text"]) for w in ws]
        xs, ys = [0, len(text)], [ws[0]["start"], ws[-1]["end"]]
        for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                t, w = toks[blk.a + k], ws[blk.b + k]
                xs += [t[0], t[1]]; ys += [w["start"], w["end"]]
        # keep strictly monotonic pairs
        pairs = sorted(zip(xs, ys))
        mx, my, last = [], [], -1.0
        for x, y in pairs:
            if y >= last and (not mx or x > mx[-1]):
                mx.append(x); my.append(y); last = y
        return mx, my

    @staticmethod
    def _from_silences(text, dur, silences):
        n = len(text)
        vc = np.cumsum([0] + [ch.isalnum() for ch in text])
        v0, v1 = 0.0, dur
        sil = list(silences or [])
        if sil and sil[0][0] < 0.05:
            v0 = sil.pop(0)[1]
        if sil and sil[-1][1] > dur - 0.05:
            v1 = sil.pop()[0]
        xs, ys = [0, n], [v0, v1]
        punct = [m.end() for m in re.finditer(r'[.,:;!?—]["”]?', text) if m.end() < n - 1]
        if sil and punct:
            gaps = sum(e - st for st, e in sil)
            rate = max(0.1, (v1 - v0 - gaps)) / max(1, vc[n])
            V, acc = [], 0.0
            for st, e in sil:
                V.append(st - v0 - acc); acc += e - st
            E = [rate * vc[q] for q in punct]
            m, k, SKIP = len(punct), len(sil), 1.5
            f = np.full((m + 1, k + 1), np.inf); f[0, :] = np.arange(k + 1) * SKIP; f[:, 0] = 0
            for i in range(1, m + 1):
                for j in range(1, k + 1):
                    f[i, j] = min(f[i - 1, j], f[i, j - 1] + SKIP, f[i - 1, j - 1] + abs(E[i - 1] - V[j - 1]))
            i, j = m, k
            while i > 0 and j > 0:
                if f[i, j] == f[i - 1, j - 1] + abs(E[i - 1] - V[j - 1]):
                    xs += [punct[i - 1], punct[i - 1] + 1]; ys += [sil[j - 1][0], sil[j - 1][1]]
                    i, j = i - 1, j - 1
                elif f[i, j] == f[i - 1, j]:
                    i -= 1
                else:
                    j -= 1
        return xs, ys

    def at(self, idx):
        return float(np.interp(idx, self.xs, self.ys))


class Ctx:
    """Per-scene context: cue(key) -> seconds into the scene where the narrator says that phrase."""
    def __init__(self, scene, timing, dur, lead, ui):
        self.sc, self.text, self.timing, self.dur, self.lead, self.ui = scene, scene["narration"], timing, dur, lead, ui

    def cue(self, key, offset=-0.18):
        phrase = self.sc.get("cues", {}).get(key, key)
        i = self.text.find(phrase)
        if i < 0:
            raise KeyError(f"cue '{key}' -> '{phrase}' not in narration")
        return max(0.0, self.lead + self.timing.at(i) + offset)

    def s(self, key):
        return self.ui[key]


# ------------------------------------------------------------------ scenes
def chapter_card(cv, c, t):
    ch = c.sc.get("chapter")
    if not ch:
        return 0
    end = min(c.lead + 1.0, 3.0)
    if t > end + 0.05:
        return end
    out = ease_io((t - (end - 0.35)) / 0.35)
    a = 1 - out
    dx = -90 * out
    num = text_img(ch["num"], "mono", 240, YEL)
    reveal(cv, num, 150 + dx, 190, ease_expo((t - 0.12) / 0.5), a)
    kick = text_img(ch["kicker"], "semi", 38, DIM, tracking=3)
    paste(cv, kick, (160 + dx - 40 * (1 - ease_out((t - 0.3) / 0.5)), 520), a * ease_out((t - 0.3) / 0.5))
    size = fit_size(ch["title"], "black", 150, 1620)
    title = text_img(ch["title"], "black", size, INK)
    reveal(cv, title, 150 + dx, 580, ease_expo((t - 0.38) / 0.55), a)
    bar = ease_expo((t - 0.2) / 0.7)
    rect(cv, (150 + dx, 590 + title.height + 18, 150 + dx + (title.width + 40) * bar, 590 + title.height + 30), YEL, a)
    return end


def scene1(cv, c, t):
    S = c.s
    s1 = c.cue("district")
    a = env(t, 0, s1 + 0.4, 0.01, 0.4)
    if a:
        date = "18.11.2026"
        n = int(clamp((t - 0.25) / 1.1) * len(date) + 0.999) if t > 0.25 else 0
        ti = text_img(date, "mono", 230, INK)
        part = text_img(date[:n], "mono", 230, INK) if n else None
        x = W / 2 - ti.width / 2
        if part:
            paste(cv, part, (x, H / 2 - ti.height / 2 - 30), a)
        cw = part.width if part else 0
        if (t * 2) % 1 < 0.6:
            rect(cv, (x + cw + 8, H / 2 - 110, x + cw + 26, H / 2 + 90), YEL, a)
        q = ease_out((t - 1.4) / 0.6)
        paste(cv, text_img(S("s1_date_sub"), "semi", 40, DIM, tracking=6), (x + 6, H / 2 + 120), a * q)
        rect(cv, (x + 6, H / 2 + 186, x + 6 + 520 * ease_expo((t - 1.6) / 0.8), H / 2 + 190), YEL, a)
    s3 = c.cue("twelve")
    box = card(cv, "image15.jpg", W / 2, H / 2 - 30, 1500, 820, s1, s3 + 0.3, t, drift=(0.7, 0.4))
    if s1 <= t <= s3 + 0.3:
        a = env(t, s1 + 0.5, s3 + 0.3)
        callout(cv, pt(box, 0.36, 0.45), (pt(box, 0.36, 0.45)[0] + 300, 250), S("s1_callout"), s1 + 1.2, t, a, 30)
        label(cv, S("s1_lower"), (230, 800), s1 + 0.6, t, 40, alpha=a)
        q = ease_out((t - s1 - 0.9) / 0.5)
        sub = text_img(S("s1_lower_sub"), "semi", 32, INK)
        rect(cv, (230, 868, 230 + (sub.width + 32) * q, 924), BG, a * 0.92)
        subline(cv, S("s1_lower_sub"), (246, 874), s1 + 0.9, t, 32, INK, a)
    s4 = c.cue("idea")
    a = env(t, s3, s4 + 0.2)
    if a:
        n = int(round(12 * ease_out((t - s3) / 1.4)))
        paste(cv, text_img(f"{n:02d}", "black", 330, YEL), (180, 250), a)
        reveal(cv, text_img(S("s1_booths"), "black", 110, INK, tracking=4), 200, 600, ease_expo((t - s3 - 0.3) / 0.5), a)
        cats = ["JKN", "Hospital Yan", "Hospital Yan", "Hospital Yan", "Hospital Yan", "Komuniti", "Hospital Yan",
                "Komuniti", "Agensi Luar", "Komuniti", "Komuniti", "Komuniti"]
        for k, cat in enumerate(cats):
            p = ease_back((t - s3 - 0.4 - k * 0.1) / 0.45)
            gx, gy = 900 + (k % 6) * 150, 300 + (k // 6) * 150
            r = 52 * max(0, p)
            if r > 1:
                layer = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
                ImageDraw.Draw(layer).ellipse((70 - r, 70 - r, 70 + r, 70 + r), fill=CAT_COL[cat] + (255,))
                paste(cv, layer, (gx - 70, gy - 70), a)
                num = text_img(str(k + 1), "black", 38, BG)
                paste(cv, num, (gx - num.width / 2, gy - 28), a * clamp(p))
        keys = ["Hospital Yan", "JKN", "Agensi Luar", "Komuniti"]
        for k, (nm, ct) in enumerate(S("s1_legend")):
            p = ease_out((t - s3 - 1.8 - k * 0.15) / 0.4)
            y = 640 + k * 62
            rect(cv, (900, y + 8, 930, y + 38), CAT_COL[keys[k]], a * p)
            paste(cv, text_img(f"{nm}  ×{ct}", "semi", 36, INK), (950 - 20 * (1 - p), y), a * p)
    a = env(t, s4, c.dur + 1, 0.3, 0.01)
    if a:
        kinetic(cv, S("s1_title1"), W / 2, H / 2 - 100, s4, t, 110, INK, alpha=a, max_w=1480)
        boxes = kinetic(cv, S("s1_title2"), W / 2, H / 2 + 50, s4 + 0.3, t, 120, INK, draw=False, max_w=1480)
        highlighter(cv, boxes[S("s1_title_hl")][1:], s4 + 1.0, t, RED, 0.9 * a)
        kinetic(cv, S("s1_title2"), W / 2, H / 2 + 50, s4 + 0.3, t, 120, INK, alpha=a, max_w=1480)
        ti = text_img(S("s1_classroom"), "semi", 40, YEL)
        paste(cv, ti, (W / 2 - ti.width / 2, H / 2 + 190), a * ease_out((t - c.cue("classroom")) / 0.6))


def scene2(cv, c, t):
    S = c.s
    t0 = chapter_card(cv, c, t)
    s1 = max(c.cue("falls"), t0 - 0.25)
    s2, s3, s4, s5, s6 = c.cue("toss"), c.cue("mystery"), c.cue("archery"), c.cue("mascot"), c.cue("play")
    b1 = card(cv, "image12.jpg", 620, 540, 600, 680, s1, s2 + 0.3, t, drift=(0.5, 0.7))
    card(cv, "image3.jpg", 1160, 560, 520, 640, s1 + 0.35, s2 + 0.3, t, drift=(0.4, 0.6))
    a = env(t, s1, s2 + 0.3)
    if a:
        label(cv, "SPINNING WHEEL", (1440, 300), s1 + 0.5, t, 44, alpha=a)
        subline(cv, S("s2_wheel_sub"), (1446, 380), s1 + 0.8, t, 30, INK, a)
        callout(cv, pt(b1, 0.72, 0.5), (pt(b1, 0.72, 0.5)[0] - 380, 165), S("s2_wheel_callout"), s1 + 1.4, t, a, 28)
    b2 = card(cv, "image2.jpg", 640, 520, 560, 800, s2, s3 + 0.3, t, drift=(0.5, 0.3))
    a = env(t, s2, s3 + 0.3)
    if a:
        label(cv, "CONE & PEARL", (1060, 330), s2 + 0.3, t, 56, alpha=a)
        for k, (big, small) in enumerate(S("s2_cone_n")):
            p = ease_back((t - s2 - 0.8 - k * 0.3) / 0.5)
            im = text_img(big, "black", 200, YEL)
            reveal(cv, im, 1070 + k * 330, 450, clamp(p), a)
            paste(cv, text_img(small, "semi", 40, INK), (1080 + k * 330, 690), a * clamp(p))
        callout(cv, pt(b2, 0.62, 0.22), (pt(b2, 0.62, 0.22)[0] + 260, 160), S("s2_cone_callout"), s2 + 1.2, t, a, 28)
    sb = c.cue("wrong")
    card(cv, "image14.png", 600, 500, 700, 700, s3, s4 + 0.3, t, zoom=0.16)
    a = env(t, s3, s4 + 0.3)
    if a:
        label(cv, "MYSTERY BOX", (1040, 300), s3 + 0.3, t, 60, alpha=a)
        subline(cv, S("s2_staff"), (1046, 410), c.cue("staff"), t, 34, INK, a)
        p = ease_expo((t - sb) / 0.45)
        if p:
            rect(cv, (1040, 500, 1040 + 780 * p, 640), RED, a)
            paste(cv, text_img(S("s2_wrong1"), "semi", 36, INK), (1070, 515), a * p)
            reveal(cv, text_img(S("s2_wrong2"), "black", 56, INK), 1070, 560, ease_expo((t - sb - 0.15) / 0.4), a)
    card(cv, "image28.png", 560, 520, 760, 780, s4, s5 + 0.3, t, drift=(0.5, 0.2))
    a = env(t, s4, s5 + 0.3)
    if a:
        label(cv, "AMS ARCHERY", (1020, 320), s4 + 0.3, t, 60, alpha=a)
        y = 470
        reveal(cv, text_img("ANTIBIOTIC", "black", 64, INK), 1030, y, ease_expo((t - s4 - 0.9) / 0.5), a)
        line(cv, (1030, y + 120), (1650, y + 120), ease_io((t - s4 - 1.2) / 0.6), YEL, 8, a)
        tri = ease_io((t - s4 - 1.2) / 0.6)
        if tri > 0:
            hx = 1030 + 620 * tri
            layer = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            ImageDraw.Draw(layer).polygon([(0, 0), (40, 20), (0, 40)], fill=YEL + (255,))
            paste(cv, layer, (hx - 20, y + 100), a)
        reveal(cv, text_img("BACTERIA", "black", 64, YEL), 1400, y + 150, ease_expo((t - s4 - 1.8) / 0.45), a)
        subline(cv, "World AMR Awareness Week · 18–24 Nov 2026", (1030, 760), s4 + 1.2, t, 30, DIM, a)
    card(cv, "image16.png", 560, 520, 620, 900, s5, c.dur + 1, t, drift=(0.5, 0.8))
    a = env(t, s5, c.dur + 1, 0.4, 0.01)
    if a:
        label(cv, "KENALI UBAT ANDA · 5B", (980, 210), s5 + 0.3, t, 48, alpha=a)
        span = max(0.5, (s6 - s5 - 1.0))
        for k, (ms, en) in enumerate(S("s2_5b_items")):
            tk = s5 + 0.8 + span * k / 5
            p = ease_back((t - tk) / 0.4)
            y = 330 + k * 105
            if p > 0:
                rect(cv, (990, y + 10, 990 + 60 * clamp(p), y + 70), YEL, a)
                paste(cv, text_img("B", "black", 44, BG), (1006, y + 12), a * clamp(p))
            reveal(cv, text_img(ms, "black", 52, INK), 1080, y + 2, ease_expo((t - tk) / 0.45), a)
            paste(cv, text_img(en, "med", 28, DIM), (1082, y + 66), a * ease_out((t - tk - 0.2) / 0.4))
        reveal(cv, text_img(S("s2_play"), "black", 56, YEL), 990, 880, ease_expo((t - s6) / 0.5), a)


def scene3(cv, c, t):
    S = c.s
    t0 = chapter_card(cv, c, t)
    s1 = max(c.cue("falls"), t0 - 0.2)
    s2, s3, s4 = c.cue("er"), c.cue("ipc"), c.cue("medsafe")
    a = env(t, t0 - 0.1, s2 + 0.3)
    if a:
        kinetic(cv, S("s3_head"), W / 2, 120, t0 - 0.1, t, 54, INK, "black", a, 0.035)
    b1 = card(cv, "image25.jpg", 560, 470, 760, 560, s1, s2 + 0.3, t, drift=(0.3, 0.5))
    b2 = card(cv, "image6.jpg", 1340, 490, 720, 580, s1 + 0.5, s2 + 0.3, t, drift=(0.6, 0.4))
    a = env(t, s1, s2 + 0.3)
    if a:
        label(cv, "PATIENT FALL-RISK ASSESSMENT", (200, 800), s1 + 0.9, t, 46, alpha=a)
        subline(cv, S("s3_fall_sub"), (210, 890), s1 + 1.3, t, 30, INK, a)
        ta = c.cue("aids")
        callout(cv, pt(b1, 0.62, 0.84), (pt(b1, 0.62, 0.84)[0] - 60, 778), S("s3_call_aids"), ta, t, a, 26)
        tr = c.cue("risk")
        callout(cv, pt(b2, 0.42, 0.42), (pt(b2, 0.42, 0.42)[0] + 300, 820), S("s3_call_risk"), tr, t, a, 26)
    a = env(t, s2, s3 + 0.3)
    if a:
        label(cv, "EMERGENCY & TRAUMA UNIT · HANDS-ON", (160, 180), s2 + 0.1, t, 40, alpha=a)
        for k, (w, key) in enumerate([("CPR", None), ("CHOKING", "choking"), ("AED", "aed")]):
            tk = s2 + 0.6 if key is None else c.cue(key)
            p = ease_expo((t - tk) / 0.45)
            y = 330 + k * 200
            paste(cv, text_img(f"0{k+1}", "mono", 70, YEL), (170, y + 40), a * p)
            reveal(cv, text_img(w, "black", 160, INK), 330, y, p, a)
        reveal(cv, text_img(S("s3_hands"), "semi", 44, YEL), 720, 790, ease_expo((t - c.cue("hands")) / 0.5), a)
    card(cv, "image4.jpg", 640, 540, 760, 800, s3, s4 + 0.3, t, zoom=0.12)
    a = env(t, s3, s4 + 0.3)
    if a:
        label(cv, "INFECTION PREVENTION", (1120, 380), s3 + 0.2, t, 50, alpha=a)
        label(cv, "& CONTROL", (1120, 460), s3 + 0.4, t, 50, alpha=a)
        reveal(cv, text_img("HAND HYGIENE", "black", 64, INK), 1126, 570, ease_expo((t - c.cue("hygiene")) / 0.5), a)
        subline(cv, "Clean hands. Safe lives.", (1126, 660), s3 + 0.9, t, 36, DIM, a)
    card(cv, "image11.png", 520, 540, 620, 900, s4, c.dur + 1, t, drift=(0.5, 0.9))
    a = env(t, s4, c.dur + 1, 0.4, 0.01)
    if a:
        label(cv, "#MedSafetyWeek MALAYSIA 2026", (930, 260), s4 + 0.2, t, 40, alpha=a)
        sq = c.cue("side")
        kinetic(cv, "“Kesan sampingan ubat —", 1360, 470, sq, t, 58, INK, "black", a, 0.05, max_w=860)
        boxes = kinetic(cv, "jangan abai, laporkan!”", 1360, 560, c.cue("report"), t, 70, YEL, draw=False, max_w=860)
        highlighter(cv, boxes[-1][1:], c.cue("report") + 0.5, t, RED, 0.85 * a)
        kinetic(cv, "jangan abai, laporkan!”", 1360, 560, c.cue("report"), t, 70, YEL, "black", a, 0.05, max_w=860)
        paste(cv, text_img(S("s3_med_sub"), "semi", 30, INK), (960, 660), a * ease_out((t - c.cue("report") - 0.6) / 0.5))


def scene4(cv, c, t):
    S = c.s
    t0 = chapter_card(cv, c, t)
    s_route = c.cue("route")
    a = env(t, t0 - 0.15, s_route + 0.3)
    if a:
        cx, cy, R = W / 2, H / 2 - 10, 300
        kinetic(cv, S("s4_head"), W / 2, 100, t0 - 0.15, t, 50, INK, "black", a, 0.035)
        partners = S("s4_partners")
        n = len(partners)
        pts = [(cx + R * 1.45 * math.cos(-math.pi / 2 + 2 * math.pi * k / n),
                cy + R * math.sin(-math.pi / 2 + 2 * math.pi * k / n)) for k in range(n)]
        for k, (lg, nm, key) in enumerate(partners):
            line(cv, (cx, cy), pts[k], ease_io((t - c.cue(key)) / 0.5), YEL, 4, a * 0.8)
        p = ease_back((t - t0 + 0.15) / 0.6)
        if p > 0:
            r = int(140 * p)
            hub = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
            ImageDraw.Draw(hub).ellipse((150 - r, 150 - r, 150 + r, 150 + r), fill=YEL + (255,))
            paste(cv, hub, (cx - 150, cy - 150), a)
            for k, wtxt in enumerate(["HOSPITAL", "YAN"]):
                ti = text_img(wtxt, "black", 54, BG)
                paste(cv, ti, (cx - ti.width / 2, cy - 64 + k * 62), a * clamp(p))
        for k, (lg, nm, key) in enumerate(partners):
            tk = c.cue(key) + 0.3
            x, y = pts[k]
            logo_disc(cv, lg, x, y, 78, tk, t, a)
            ti = text_img(nm, "semi", 28, INK)
            paste(cv, ti, (x - ti.width / 2, y + 92), a * ease_out((t - tk - 0.2) / 0.4))
    a = env(t, s_route, c.dur + 1, 0.4, 0.01)
    if a:
        label(cv, "INTER-STATE COLLABORATION", (160, 160), s_route + 0.1, t, 44, alpha=a)
        p0, p1 = (300, 340), (640, 800)
        for (q, nm, tk) in [(p0, "KEDAH · Hospital Yan", s_route + 0.3), (p1, "NEGERI SEMBILAN · CRC", s_route + 1.6)]:
            pq = ease_back((t - tk) / 0.45)
            if pq > 0:
                ring = _dot(16, 6)
                paste(cv, ring, (q[0] - ring.width / 2, q[1] - ring.height / 2), a * clamp(pq))
            reveal(cv, text_img(nm, "black", 40, INK), q[0] + 50, q[1] - 28, ease_expo((t - tk - 0.1) / 0.45), a)
        line(cv, p0, p1, ease_io((t - s_route - 0.5) / 1.1), YEL, 6, a, dash=18)
        card(cv, "image23.jpg", 1580, 500, 500, 640, s_route + 1.0, c.dur + 1, t, drift=(0.5, 0.5))
        subline(cv, "Nursing Research Team, Clinical Research Centre", (1150, 860), s_route + 1.8, t, 28, INK, a)


def scene5(cv, c, t):
    S = c.s
    t0 = chapter_card(cv, c, t)
    s1 = max(c.cue("open"), t0 - 0.2)
    s2, s3, s4 = c.cue("marks"), c.cue("lead"), c.cue("end")
    a = env(t, t0 - 0.2, s1 + 0.2)
    if a:
        kinetic(cv, S("s5_head"), W / 2, H / 2 + 220, t0 - 0.2, t, 72, INK, "black", a, 0.05)
    card(cv, "image28.png", 560, 540, 760, 820, s1, s3 + 0.3, t, drift=(0.5, 0.3))
    a = env(t, s1, s3 + 0.3)
    if a:
        label(cv, "18–24 NOV  ·  WAAW 2026", (1020, 300), s1 + 0.3, t, 50, alpha=a)
        subline(cv, "One Health, One Action: Prevent AMR Now", (1026, 400), s1 + 0.7, t, 36, INK, a)
        reveal(cv, text_img("PATIENT SAFETY", "black", 100, INK), 1020, 520, ease_expo((t - s2) / 0.5), a)
        reveal(cv, text_img("DAY 2026", "black", 100, YEL), 1020, 630, ease_expo((t - s2 - 0.15) / 0.5), a)
        subline(cv, S("s5_psd_sub"), (1026, 770), s2 + 0.6, t, 34, DIM, a)
    card(cv, "image15.jpg", W / 2, H / 2 - 60, 1600, 860, s3, s4 + 0.3, t, zoom=0.08, drift=(0.3, 0.5))
    a = env(t, s3, s4 + 0.3)
    if a:
        p = ease_expo((t - s3 - 0.4) / 0.55)
        rect(cv, (160, 780, 160 + 1100 * p, 930), BG, a * 0.92)
        rect(cv, (160, 780, 172, 930), YEL, a * p)
        reveal(cv, text_img("Dr. Tengku Nurshahril", "black", 64, INK), 200, 790, ease_expo((t - s3 - 0.55) / 0.5), a)
        paste(cv, text_img(S("s5_role"), "semi", 36, YEL), (204, 870), a * ease_out((t - s3 - 0.8) / 0.4))
        q = ease_expo((t - c.cue("walks")) / 0.55)
        if q:
            qt = text_img(S("s5_quote"), "black", fit_size(S("s5_quote"), "black", 46, 1500), BG)
            rect(cv, (160, 160, 160 + (qt.width + 60) * q, 260), YEL, a)
            reveal(cv, qt, 190, 182, ease_expo((t - c.cue("walks") - 0.15) / 0.5), a)
    a = env(t, s4, c.dur + 1, 0.4, 0.01)
    if a:
        kinetic(cv, "HOSPITAL YAN", W / 2, H / 2 - 160, s4, t, 150, INK, alpha=a)
        kinetic(cv, "COMMUNITY CARNIVAL 2026", W / 2, H / 2, s4 + 0.25, t, 96, YEL, alpha=a)
        q = ease_expo((t - s4 - 0.6) / 0.7)
        rect(cv, (W / 2 - 420 * q, H / 2 + 90, W / 2 + 420 * q, H / 2 + 96), INK, a)
        tg = c.cue("together")
        ti = text_img(S("s5_tag"), "semi", 52, INK)
        reveal(cv, ti, W / 2 - ti.width / 2, H / 2 + 130, ease_expo((t - tg) / 0.6), a)
        ti = text_img("18 NOVEMBER 2026  ·  HOSPITAL YAN, KEDAH", "mono", 30, DIM)
        paste(cv, ti, (W / 2 - ti.width / 2, H / 2 + 230), a * ease_out((t - tg - 0.5) / 0.6))


SCENES = [scene1, scene2, scene3, scene4, scene5]


# ------------------------------------------------------------------- frame
NBSP = " "

def caption_chunks(text, maxlen=62):
    words, out, cur = text.split(" "), [], ""
    for w in words:
        if not w:
            continue
        if len(cur) + len(w) + 1 > maxlen or (cur and cur[-1] in ".:!?—"):
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip(" ")
    if cur:
        out.append(cur)
    return out

@lru_cache(maxsize=1)
def wipe_band():
    """Skewed yellow/navy band used for chapter wipes (wider than the frame)."""
    skew = 260
    bw = W + skew * 2 + 160
    im = Image.new("RGBA", (bw, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(skew, 0), (bw - 80, 0), (bw - 80 - skew, H), (0, H)], fill=YEL + (255,))
    d.polygon([(bw - 80, 0), (bw, 0), (bw - skew, H), (bw - 80 - skew, H)], fill=NAVY + (255,))
    return im

class Plan:
    def __init__(self, cfg, durations, leads, timings, captions):
        self.cfg, self.durs, self.leads, self.timings, self.captions = cfg, durations, leads, timings, captions
        self.ui = cfg["ui"]
        self.starts = np.cumsum([0] + durations[:-1]).tolist()
        self.total = sum(durations)
        subs = cfg.get("caption_subs", [])
        self.caps = []
        for i, sc in enumerate(cfg["scenes"]):
            text = sc["narration"]
            prot = text
            for src, _ in subs:
                prot = prot.replace(src, src.replace(" ", NBSP))
            pos = 0
            for ch in caption_chunks(prot):
                a_i = prot.find(ch, pos); pos = a_i + len(ch)
                disp = ch.replace(NBSP, " ")
                for src, dst in subs:
                    disp = disp.replace(src, dst)
                a = leads[i] + timings[i].at(a_i)
                b = leads[i] + timings[i].at(pos)
                self.caps.append((self.starts[i] + a - 0.05, self.starts[i] + b + 0.2, disp))
        # no overlaps: each caption ends when the next starts
        for k in range(len(self.caps) - 1):
            a, b, s = self.caps[k]
            self.caps[k] = (a, min(b, self.caps[k + 1][0]), s)

    def scene_at(self, t):
        return max(k for k in range(len(self.starts)) if self.starts[k] <= t + 1e-9)

    def frame(self, t):
        cv = backdrop().copy()
        i = self.scene_at(t)
        sc = self.cfg["scenes"][i]
        lt = t - self.starts[i]
        SCENES[i](cv, Ctx(sc, self.timings[i], self.durs[i], self.leads[i], self.ui), lt)
        # motion cut: a quick settle-in zoom after each wipe
        if i > 0 and lt < 0.6:
            z = 1 + 0.05 * (1 - ease_out(lt / 0.6))
            zw, zh = int(W / z), int(H / z)
            cv = cv.resize((W, H), Image.BILINEAR, box=((W - zw) / 2, (H - zh) / 2, (W + zw) / 2, (H + zh) / 2))
        # chrome
        chip = text_img(f"{i+1:02d}  {sc['title'].upper()}", "mono", 22, DIM)
        paste(cv, chip, (60, 44), 0.9)
        rect(cv, (60, 76, 60 + 40 * ease_expo(lt / 0.8), 79), YEL, 0.9)
        rect(cv, (0, H - 10, W, H), (40, 41, 46))
        rect(cv, (0, H - 10, W * t / self.total, H), YEL)
        for s in self.starts[1:]:
            rect(cv, (W * s / self.total - 2, H - 16, W * s / self.total + 2, H), INK, 0.8)
        if self.captions:
            for a, b, txt in self.caps:
                if a <= t < b:
                    ti = text_img(txt, "semi", 34, INK)
                    p = ease_out((t - a) / 0.18)
                    x, y = W / 2 - ti.width / 2, H - 92 + 8 * (1 - p)
                    rect(cv, (x - 18, y - 8, x + ti.width + 18, y + ti.height + 4), (0, 0, 0), 0.62 * p)
                    paste(cv, ti, (x, y), p)
                    break
        cv.alpha_composite(vignette())
        cv.alpha_composite(grain_layers()[int(t * 12) % 6])
        # chapter wipe: skewed band sweeps across, covering the cut
        for s in self.starts[1:]:
            u = (t - (s - WIPE)) / (2 * WIPE)
            if 0 <= u <= 1:
                band = wipe_band()
                x = -band.width + (W + band.width) * ease_io(u)
                paste(cv, band, (x, 0))
        # open from black, close to black
        if t < 0.5:
            rect(cv, (0, 0, W, H), (0, 0, 0), 1 - t / 0.5)
        if t > self.total - 1.2:
            rect(cv, (0, 0, W, H), (0, 0, 0), clamp((t - (self.total - 1.2)) / 1.2))
        return cv.convert("RGB")


# ------------------------------------------------------------------- audio
def decode(path, filters="", sr=SR):
    af = (filters + "," if filters else "") + f"aresample={sr}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-af", af, "-ac", "2", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()

def onepole(x, coef):
    """y[n] = y[n-1] + coef*(x[n]-y[n-1]); coef may be an array (time-varying)."""
    y = np.empty_like(x)
    acc = 0.0
    c = np.broadcast_to(coef, x.shape)
    for n in range(len(x)):
        acc += c[n] * (x[n] - acc)
        y[n] = acc
    return y

def sfx_whoosh(dur=0.85, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    tt = np.linspace(0, 1, n)
    shape = np.sin(np.pi * tt ** 0.8) ** 2
    fc = 250 + 5200 * shape
    coef = 1 - np.exp(-2 * np.pi * fc / SR)
    out = []
    for ch in range(2):
        nz = rng.standard_normal(n)
        lp = onepole(nz, coef)
        hp = lp - onepole(lp, 1 - np.exp(-2 * np.pi * 180 / SR))
        out.append(hp * shape)
    pan = tt
    L, R = out[0] * np.sqrt(1 - pan), out[1] * np.sqrt(pan)
    s = np.stack([L, R], 1)
    return (s / np.abs(s).max() * 0.32).astype(np.float32)

def sfx_impact(seed=2):
    rng = np.random.default_rng(seed)
    n = int(1.4 * SR)
    tt = np.arange(n) / SR
    f = 38 + 70 * np.exp(-tt / 0.06)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.38)
    click = rng.standard_normal(n) * np.exp(-tt / 0.006)
    click = click - onepole(click, 1 - np.exp(-2 * np.pi * 900 / SR))
    tail = onepole(rng.standard_normal(n), 1 - np.exp(-2 * np.pi * 1400 / SR)) * np.exp(-tt / 0.5) * 0.35
    m = boom + 0.25 * click + tail
    s = np.stack([m, m], 1)
    return (s / np.abs(s).max() * 0.42).astype(np.float32)

def sfx_tick(seed=3):
    rng = np.random.default_rng(seed)
    n = int(0.03 * SR)
    tt = np.arange(n) / SR
    x = rng.standard_normal(n) * np.exp(-tt / 0.0025)
    x = np.diff(np.concatenate([[0], x]))
    s = np.stack([x, x], 1)
    return (s / np.abs(s).max() * 0.10).astype(np.float32)

def add(buf, clip, t):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(clip))
    buf[max(i, 0):j] += clip[max(0, -i):j - i]

def build_mix(plan, cfg, audio, tempo, music_path, wav_out):
    total_n = int(math.ceil(plan.total * SR))
    voice = np.zeros((total_n, 2), np.float32)
    for i, p in enumerate(audio):
        if p:
            v = decode(p, f"atempo={tempo:.4f}" if abs(tempo - 1) > 1e-3 else "")
            add(voice, v, plan.starts[i] + plan.leads[i])
    # voice activity (10 ms hops), pre-roll 80 ms, hold 220 ms
    hop = SR // 100
    nh = total_n // hop
    rms = np.sqrt((voice[:nh * hop].mean(1) ** 2).reshape(nh, hop).mean(1) + 1e-12)
    act = rms > 10 ** (-42 / 20)
    k = np.ones(22 + 8, bool)
    act = np.convolve(act, k, "same")[: nh] > 0
    act = np.roll(act, -4)
    G_SPEECH, G_PAUSE, G_SWELL = cfg.get("music_gains", [0.11, 0.33, 0.62])
    tgt = np.where(act, G_SPEECH, G_PAUSE).astype(np.float64)
    tt = np.arange(nh) / 100.0
    # swells: intro until the first word, every chapter transition until its narrator starts, outro
    windows = [(0, plan.leads[0] + 0.1)]
    windows += [(s - 1.2, s + plan.leads[k] - 0.1) for k, s in enumerate(plan.starts) if k > 0]
    last_voice = plan.starts[-1] + plan.leads[-1] + plan.speech[-1]
    windows.append((last_voice + 0.2, plan.total))
    for a, b in windows:
        m = (tt >= a) & (tt <= b) & ~act
        tgt[m] = G_SWELL
    # asymmetric smoothing: duck fast (90 ms), recover slower (450 ms)
    g = np.empty_like(tgt)
    acc = tgt[0]
    da, ua = 1 - math.exp(-1 / (100 * 0.09)), 1 - math.exp(-1 / (100 * 0.45))
    for n in range(nh):
        acc += (da if tgt[n] < acc else ua) * (tgt[n] - acc)
        g[n] = acc
    gain = np.interp(np.arange(total_n) / SR, tt, g).astype(np.float32)

    mix = voice * 1.0
    if music_path:
        mu = decode(music_path, "loudnorm=I=-16:TP=-2:LRA=11")
        need = plan.total
        have = len(mu) / SR
        f = have / need
        if 0.9 < f < 1.1 and abs(f - 1) > 0.003:
            mu = decode(music_path, f"loudnorm=I=-16:TP=-2:LRA=11,atempo={f:.5f}")
        if len(mu) < total_n:
            mu = np.concatenate([mu, np.zeros((total_n - len(mu), 2), np.float32)])
        mu = mu[:total_n]
        fo = int(2.5 * SR)
        mu[-fo:] *= np.linspace(1, 0, fo)[:, None]
        mix += mu * gain[:, None]
        print(f"[audio] music {have:.1f}s fitted to {need:.1f}s (x{f:.3f})", file=sys.stderr)
    # sfx
    wh, im, tk = sfx_whoosh(), sfx_impact(), sfx_tick()
    for k, s in enumerate(plan.starts):
        if k > 0:
            add(mix, wh, s - WIPE - 0.25)
            add(mix, im * 0.8, s + 0.1)
    add(mix, im * 0.7, 0.2)
    for j in range(10):
        add(mix, tk, 0.25 + 1.1 * (j + 0.5) / 10)
    add(mix, im * 0.9, plan.starts[-1] + plan.ui_end_time)
    raw = os.path.splitext(wav_out)[0] + ".raw"
    mix.astype(np.float32).tofile(raw)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", raw,
                    "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-ar", str(SR), wav_out], check=True)
    os.remove(raw)


# ------------------------------------------------------------------- main
def silences(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "silencedetect=noise=-35dB:d=0.15",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", err)]
    en = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", err)]
    return list(zip(st, en))

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    return float(out.stdout.strip())

_PLAN = None
def _init(plan):
    global _PLAN
    _PLAN = plan
def _render(args):
    t, size = args
    im = _PLAN.frame(t)
    if size != (W, H):
        im = im.resize(size, Image.BILINEAR)
    return im.tobytes()

def load_plan(cfg, captions=True):
    tempo = cfg.get("tempo", 1.0)
    leads, tails = cfg.get("leads", [0.35] * 5), cfg.get("tails", [0.7] * 5)
    durs, timings, audio, speech = [], [], [], []
    for i, sc in enumerate(cfg["scenes"]):
        p = os.path.join(ROOT, sc["audio"])
        wj = os.path.splitext(p)[0] + ".words.json"
        if os.path.exists(p):
            d = probe(p) / tempo; audio.append(p)
            words = json.load(open(wj))["words"] if os.path.exists(wj) else None
            timings.append(Timing(sc["narration"], d, None if words else silences(p), words, tempo))
        else:
            d = sc.get("target_seconds", 30) - leads[i] - tails[i]; audio.append(None)
            timings.append(Timing(sc["narration"], d))
            print(f"[warn] {sc['audio']} missing — using {d:.1f}s of silence", file=sys.stderr)
        speech.append(d)
        durs.append(leads[i] + d + tails[i])
    plan = Plan(cfg, durs, leads, timings, captions)
    plan.speech = speech
    c5 = Ctx(cfg["scenes"][-1], timings[-1], durs[-1], leads[-1], cfg["ui"])
    plan.ui_end_time = c5.cue("end") + 0.05
    return plan, audio, tempo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "ms"])
    ap.add_argument("--out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview", action="store_true", help="960x540 @ 15fps, fast encode")
    ap.add_argument("--still", type=float, action="append", help="write a PNG at this time (s) and exit")
    ap.add_argument("--music", help="music bed (overrides the config's)")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--audio-only", action="store_true", help="write only the mixed WAV next to --out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "scenes.json" if a.lang == "en" else f"scenes_{a.lang}.json")))
    out = a.out or os.path.join(ROOT, cfg.get("out", "out/hyan_carnival_2026.mp4"))
    plan, audio, tempo = load_plan(cfg, not a.no_captions)
    print(f"[plan] {a.lang}: scenes {', '.join(f'{d:.1f}s' for d in plan.durs)}  total {plan.total:.1f}s", file=sys.stderr)

    if a.still:
        os.makedirs(os.path.join(ROOT, "out", "stills"), exist_ok=True)
        for t in a.still:
            fn = os.path.join(ROOT, "out", "stills", f"{a.lang}_t{t:06.2f}.png")
            plan.frame(t).save(fn); print(fn)
        return

    music = None if a.no_music else (a.music or (os.path.join(ROOT, cfg["music"]) if cfg.get("music") else None))
    if music and not os.path.exists(music):
        print(f"[warn] music {music} missing — rendering without music", file=sys.stderr)
        music = None
    fps = 15 if a.preview else a.fps
    size = (960, 540) if a.preview else (W, H)
    nframes = int(math.ceil(plan.total * fps))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="hyan_")
    try:
        wav = os.path.join(tmp, "mix.wav")
        build_mix(plan, cfg, audio, tempo, music, wav)
        if a.audio_only:
            dst = os.path.splitext(out)[0] + ".wav"
            shutil.copy(wav, dst); print(f"[done] {dst}", file=sys.stderr)
            return
        enc = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "veryfast" if a.preview else "medium", "-crf", "28" if a.preview else "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
        ff = subprocess.Popen(enc, stdin=subprocess.PIPE)
        with Pool(a.jobs, initializer=_init, initargs=(plan,)) as pool:
            # Bounded batches: Pool.imap buffers results without limit, and frames
            # render faster than x264 encodes them, which exhausts memory at 1080p.
            batch = a.jobs * 8
            for b0 in range(0, nframes, batch):
                jobs = [(f / fps, size) for f in range(b0, min(nframes, b0 + batch))]
                for buf in pool.imap(_render, jobs, chunksize=2):
                    ff.stdin.write(buf)
                print(f"\r[render] {min(nframes, b0 + batch)/fps:6.1f}s / {plan.total:.1f}s", end="", file=sys.stderr, flush=True)
        ff.stdin.close()
        if ff.wait() != 0:
            sys.exit("ffmpeg encode failed")
        print(f"\n[done] {out}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
