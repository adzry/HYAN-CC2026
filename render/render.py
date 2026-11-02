#!/usr/bin/env python3
"""Vox-style renderer for the Hospital Yan Community Carnival 2026 video.

Draws every frame with Pillow (dark theme, kinetic type, Ken Burns photo cards,
chapter cards, progress bar, captions) and pipes them into FFmpeg together with
the per-scene voiceover in ./audio. Scene lengths follow the real audio length;
if a scene's MP3 is missing, the scene's target length is used with silence.

Usage (from the repo root; normally via ./render_video.sh):
  python3 render/render.py [--out out/video.mp4] [--fps 30] [--preview]
                           [--still SECONDS] [--music bed.mp3] [--no-captions]
"""
import argparse, json, math, os, random, shutil, subprocess, sys, tempfile
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "render")
W, H = 1920, 1080
TAIL = 0.7          # breathing room after each scene's narration (s)
LEAD = 0.35         # silence before narration starts in each scene (s)

BG = (14, 15, 18)
INK = (240, 237, 228)
DIM = (150, 150, 158)
YEL = (255, 210, 63)
RED = (232, 72, 60)
TEAL = (64, 196, 180)
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

def paste(canvas, im, xy, alpha=1.0):
    if alpha <= 0.003:
        return
    x, y = int(round(xy[0])), int(round(xy[1]))
    if alpha < 0.997:
        im = im.copy()
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im.putalpha(a)
    # crop to canvas bounds
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

def label(canvas, text, xy, t_in, t, size=34, bg=YEL, fg=(12, 12, 14), alpha=1.0):
    """Yellow tag that wipes in from the left."""
    p = ease_out((t - t_in) / 0.45)
    if p <= 0:
        return
    ti = text_img(text, "black", size, fg)
    pad = int(size * 0.45)
    w, h = ti.width + pad * 2, ti.height + pad
    rect(canvas, (xy[0], xy[1], xy[0] + w * p, xy[1] + h), bg, alpha)
    if p > 0.35:
        crop = ti.crop((0, 0, max(1, int(ti.width * clamp((p - 0.35) / 0.65))), ti.height))
        paste(canvas, crop, (xy[0] + pad, xy[1] + pad * 0.35), alpha)

def subline(canvas, text, xy, t_in, t, size=30, color=INK, alpha=1.0):
    p = ease_out((t - t_in) / 0.5)
    paste(canvas, text_img(text, "semi", size, color), (xy[0] - 30 * (1 - p), xy[1]), alpha * p)

def kinetic(canvas, text, cx, cy, t_in, t, size=110, color=INK, fname="black", alpha=1.0, stagger=0.07, rise=40):
    """Word-by-word rise-and-fade headline centred on (cx, cy). Returns word boxes."""
    words = text.split(" ")
    imgs = [text_img(w, fname, size, color) for w in words]
    space = font(fname, size).getlength(" ")
    total = sum(i.width for i in imgs) + space * (len(imgs) - 1)
    x = cx - total / 2
    boxes = []
    for k, (w, im) in enumerate(zip(words, imgs)):
        p = ease_out((t - t_in - k * stagger) / 0.45)
        paste(canvas, im, (x, cy - im.height / 2 + rise * (1 - p)), alpha * p)
        boxes.append((w, x, cy - im.height / 2, im.width, im.height))
        x += im.width + space
    return boxes

def highlighter(canvas, box, t_in, t, color=YEL, alpha=0.9, dur=0.5):
    x, y, w, h = box
    p = ease_io((t - t_in) / dur)
    if p > 0:
        rect(canvas, (x - 10, y + h * 0.18, x - 10 + (w + 20) * p, y + h * 0.95), color, alpha)


# ------------------------------------------------------------- photo cards
@lru_cache(maxsize=64)
def src_img(name):
    return Image.open(os.path.join(ROOT, "images", name)).convert("RGB")

@lru_cache(maxsize=64)
def shadow(w, h, blur=28):
    pad = blur * 3
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((pad, pad + 12, pad + w, pad + h + 12), fill=(0, 0, 0, 170))
    return im.filter(ImageFilter.GaussianBlur(blur)), pad

def card(canvas, name, cx, cy, max_w, max_h, t_in, t_out, t, border=10, zoom=0.09, drift=(0.5, 0.5),
         slide=(0, 60), alpha=1.0):
    """Photo on a white paper card with Ken Burns motion, sliding in/out."""
    a = env(t, t_in, t_out, 0.45, 0.35) * alpha
    if a <= 0:
        return None
    src = src_img(name)
    sw, sh = src.size
    s = min((max_w - 2 * border) / sw, (max_h - 2 * border) / sh)
    iw, ih = int(sw * s), int(sh * s)
    life = clamp((t - t_in) / max(0.1, t_out - t_in))
    z = 1 + zoom * life
    cw, ch = sw / z, sh / z
    ox = (sw - cw) * (drift[0] if life else 0.5) * life + (sw - cw) * 0.5 * (1 - life)
    oy = (sh - ch) * (drift[1] if life else 0.5) * life + (sh - ch) * 0.5 * (1 - life)
    photo = src.resize((iw, ih), Image.BILINEAR, box=(ox, oy, ox + cw, oy + ch))
    w, h = iw + 2 * border, ih + 2 * border
    p = ease_out((t - t_in) / 0.6)
    x = cx - w / 2 + slide[0] * (1 - p)
    y = cy - h / 2 + slide[1] * (1 - p)
    sh_im, pad = shadow(w, h)
    paste(canvas, sh_im, (x - pad, y - pad), a * 0.9)
    paperc = Image.new("RGBA", (w, h), (246, 243, 236, 255))
    paperc.paste(photo, (border, border))
    paste(canvas, paperc, (x, y), a)
    return (x, y, w, h)

def logo_disc(canvas, name, cx, cy, r, t_in, t, alpha=1.0):
    p = ease_out((t - t_in) / 0.5)
    if p <= 0:
        return
    rr = max(2, int(r * (0.6 + 0.4 * p)))
    im = _logo_disc(name, rr)
    paste(canvas, im, (cx - im.width / 2, cy - im.height / 2), alpha * p)

@lru_cache(maxsize=64)
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

def line(canvas, p0, p1, prog, color=YEL, width=4, alpha=1.0, dash=0):
    prog = clamp(prog)
    if prog <= 0 or alpha <= 0:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x1 = p0[0] + (p1[0] - p0[0]) * prog
    y1 = p0[1] + (p1[1] - p0[1]) * prog
    if dash:
        L = math.hypot(x1 - p0[0], y1 - p0[1])
        n = int(L // (dash * 2)) + 1
        for k in range(n):
            a0, a1 = k * 2 * dash / max(L, 1), min(1, (k * 2 * dash + dash) / max(L, 1))
            d.line([(p0[0] + (x1 - p0[0]) * a0, p0[1] + (y1 - p0[1]) * a0),
                    (p0[0] + (x1 - p0[0]) * a1, p0[1] + (y1 - p0[1]) * a1)], fill=color + (int(255 * alpha),), width=width)
    else:
        d.line([p0, (x1, y1)], fill=color + (int(255 * alpha),), width=width)
    canvas.alpha_composite(layer)


# ------------------------------------------------------------- global layers
@lru_cache(maxsize=1)
def grain_layers():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(6):
        n = rng.integers(0, 255, (H // 2, W // 2), dtype=np.uint8)
        a = np.full_like(n, 14)
        im = Image.fromarray(np.dstack([n, n, n, a]), "RGBA").resize((W, H), Image.NEAREST)
        out.append(im)
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
    """Dark base with a faint grid, Vox-explainer style."""
    im = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 120):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 7))
    for y in range(0, H, 120):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 7))
    return im


# ------------------------------------------------------------------ scenes
class Ctx:
    """Per-scene timing: cue(phrase) -> seconds into the scene where the narrator says it."""
    def __init__(self, scene, speech, dur):
        self.text, self.speech, self.dur = scene["narration"], speech, dur

    def cue(self, phrase, offset=-0.25):
        i = self.text.find(phrase)
        if i < 0:
            raise KeyError(f"cue '{phrase}' not in narration")
        return max(0.0, LEAD + self.speech * i / len(self.text) + offset)


def chapter_card(cv, c, num, word, t):
    end = min(2.2, c.dur * 0.15)
    a = env(t, 0, end, 0.15, 0.35)
    if a <= 0:
        return end
    p = ease_out(t / 0.5)
    rect(cv, (0, H / 2 - 6, W * p, H / 2 + 6), YEL, a)
    paste(cv, text_img(num, "mono", 210, YEL), (160, H / 2 - 260 + 30 * (1 - p)), a * p)
    q = ease_out((t - 0.2) / 0.5)
    paste(cv, text_img(word, "black", 170, INK, tracking=6), (160, H / 2 + 40 + 40 * (1 - q)), a * q)
    return end


def scene1(cv, c, t):
    # 18.11.2026 typewriter
    s1 = c.cue("A district")
    a = env(t, 0, s1 + 0.4, 0.01, 0.4)
    if a:
        date = "18.11.2026"
        n = int(clamp(t / 1.4) * len(date) + 0.999)
        ti = text_img(date, "mono", 230, INK)
        part = text_img(date[:n], "mono", 230, INK)
        x = W / 2 - ti.width / 2
        paste(cv, part, (x, H / 2 - ti.height / 2 - 30), a)
        if (t * 2) % 1 < 0.6:
            rect(cv, (x + part.width + 8, H / 2 - 110, x + part.width + 26, H / 2 + 90), YEL, a)
        q = ease_out((t - 1.5) / 0.6)
        paste(cv, text_img("HOSPITAL YAN COMMUNITY CARNIVAL", "semi", 40, DIM, tracking=6), (x + 6, H / 2 + 120), a * q)
    # Photo + lower third
    s2, s3 = s1, c.cue("Twelve booths")
    card(cv, "image15.jpg", W / 2, H / 2 - 30, 1500, 820, s2, s3 + 0.3, t, drift=(0.7, 0.4))
    if s2 <= t <= s3 + 0.3:
        a = env(t, s2 + 0.5, s3 + 0.3)
        label(cv, "HOSPITAL YAN  ·  KEDAH", (230, 800), s2 + 0.6, t, 40, alpha=a)
        q = ease_out((t - s2 - 0.9) / 0.5)
        rect(cv, (230, 868, 230 + 820 * q, 924), BG, a * 0.9)
        subline(cv, "Not to admit patients — to invite a community in.", (246, 874), s2 + 0.9, t, 32, INK, a)
    # 12 booths counter + category dots
    s4 = c.cue("One idea")
    a = env(t, s3, s4 + 0.2)
    if a:
        n = int(round(12 * ease_out((t - s3) / 1.6)))
        num = text_img(f"{n:02d}", "black", 330, YEL)
        paste(cv, num, (180, 250), a)
        paste(cv, text_img("BOOTHS", "black", 110, INK, tracking=4), (200, 600), a * ease_out((t - s3 - 0.3) / 0.5))
        cats = ["JKN"] + ["Hospital Yan"] * 2 + ["Hospital Yan"] * 2 + ["Komuniti", "Hospital Yan", "Komuniti", "Agensi Luar", "Komuniti", "Komuniti", "Komuniti"]
        for k, cat in enumerate(cats):
            p = ease_out((t - s3 - 0.4 - k * 0.12) / 0.4)
            gx, gy = 900 + (k % 6) * 150, 300 + (k // 6) * 150
            r = 52 * p
            if r > 1:
                layer = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
                ImageDraw.Draw(layer).ellipse((60 - r, 60 - r, 60 + r, 60 + r), fill=CAT_COL[cat] + (255,))
                paste(cv, layer, (gx - 60, gy - 60), a)
                paste(cv, text_img(str(k + 1), "black", 38, BG), (gx - 12 - (10 if k >= 9 else 0), gy - 26), a * p)
        legend = [("Hospital Yan", "5"), ("JKN Kedah", "1"), ("Agensi Luar", "1"), ("Komuniti", "5")]
        for k, (nm, ct) in enumerate(legend):
            key = ["Hospital Yan", "JKN", "Agensi Luar", "Komuniti"][k]
            p = ease_out((t - s3 - 2.0 - k * 0.15) / 0.4)
            y = 640 + k * 62
            rect(cv, (900, y + 8, 930, y + 38), CAT_COL[key], a * p)
            paste(cv, text_img(f"{nm}  ×{ct}", "semi", 36, INK), (950, y), a * p)
    # Title statement
    a = env(t, s4, c.dur, 0.3, 0.4)
    if a:
        b1 = kinetic(cv, "THE SAFEST PATIENT IS", W / 2, H / 2 - 90, s4, t, 110, INK, alpha=a)
        boxes = kinetic(cv, "AN INFORMED ONE.", W / 2, H / 2 + 60, s4 + 0.35, t, 130, INK, alpha=0)  # layout only
        highlighter(cv, boxes[1][1:], s4 + 0.9, t, RED, 0.85 * a)
        kinetic(cv, "AN INFORMED ONE.", W / 2, H / 2 + 60, s4 + 0.35, t, 130, INK, alpha=a)
        q = ease_out((t - c.cue("And this is how")) / 0.6)
        ti = text_img("How a hospital compound becomes a classroom", "semi", 38, YEL)
        paste(cv, ti, (W / 2 - ti.width / 2, H / 2 + 200), a * q)


def scene2(cv, c, t):
    chapter_card(cv, c, "01", "PLAY", t)
    s1, s2, s3 = c.cue("At the patient-falls"), c.cue("They toss"), c.cue("Then comes the Mystery")
    s4, s5, s6 = c.cue("Next door"), c.cue("And a capsule"), c.cue("It's learning")
    # spinning wheel
    card(cv, "image12.jpg", 620, 480, 640, 760, s1, s2 + 0.3, t, drift=(0.5, 0.7))
    card(cv, "image3.jpg", 1180, 520, 560, 700, s1 + 0.35, s2 + 0.3, t, drift=(0.4, 0.6))
    a = env(t, s1, s2 + 0.3)
    if a:
        label(cv, "SPINNING WHEEL", (1440, 300), s1 + 0.5, t, 44, alpha=a)
        subline(cv, "+ the “Falling Love” booklet", (1446, 380), s1 + 0.8, t, 32, INK, a)
    # cone & pearl
    card(cv, "image2.jpg", 640, 520, 560, 800, s2, s3 + 0.3, t, drift=(0.5, 0.3))
    a = env(t, s2, s3 + 0.3)
    if a:
        label(cv, "CONE & PEARL", (1060, 330), s2 + 0.3, t, 56, alpha=a)
        for k, (big, small) in enumerate([("5", "throws"), ("3", "questions")]):
            p = ease_out((t - s2 - 0.8 - k * 0.3) / 0.5)
            paste(cv, text_img(big, "black", 200, YEL), (1070 + k * 330, 450), a * p)
            paste(cv, text_img(small, "semi", 40, INK), (1080 + k * 330, 690), a * p)
    # mystery box
    sb = c.cue("a wrong answer")
    card(cv, "image14.png", 600, 500, 700, 700, s3, s4 + 0.3, t, zoom=0.15)
    a = env(t, s3, s4 + 0.3)
    if a:
        label(cv, "MYSTERY BOX", (1040, 300), s3 + 0.3, t, 60, alpha=a)
        subline(cv, "Staff: 5 bonus fall-risk diagnosis questions", (1046, 410), c.cue("Health staff"), t, 34, INK, a)
        p = ease_out((t - sb) / 0.4)
        if p:
            rect(cv, (1040, 500, 1040 + 760 * p, 640), RED, a)
            paste(cv, text_img("Wrong answer?", "semi", 36, INK), (1070, 515), a * p)
            paste(cv, text_img("Chew a burnt biscuit.", "black", 56, INK), (1070, 560), a * p)
    # AMS archery
    card(cv, "image28.png", 560, 520, 760, 780, s4, s5 + 0.3, t, drift=(0.5, 0.2))
    a = env(t, s4, s5 + 0.3)
    if a:
        label(cv, "AMS ARCHERY", (1020, 320), s4 + 0.3, t, 60, alpha=a)
        y = 470
        p = ease_out((t - s4 - 0.9) / 0.6)
        paste(cv, text_img("ANTIBIOTIC", "black", 64, INK), (1030, y), a * p)
        line(cv, (1030, y + 120), (1650, y + 120), ease_io((t - s4 - 1.2) / 0.7), YEL, 8, a)
        if t > s4 + 1.8:
            paste(cv, text_img("➝ BACTERIA", "black", 64, YEL), (1300, y + 150), a * ease_out((t - s4 - 1.8) / 0.4))
        subline(cv, "World AMR Awareness Week · 18–24 Nov 2026", (1030, 760), s4 + 1.2, t, 30, DIM, a)
    # 5B mascot
    card(cv, "image16.png", 560, 520, 620, 900, s5, c.dur, t, drift=(0.5, 0.8))
    a = env(t, s5, c.dur, 0.45, 0.4)
    if a:
        label(cv, "KENALI UBAT ANDA · 5B", (980, 210), s5 + 0.3, t, 48, alpha=a)
        items = [("Pengguna betul", "right user"), ("Ubat betul", "right medicine"), ("Dos betul", "right dose"),
                 ("Cara betul", "right way"), ("Masa betul", "right time")]
        span = max(0.5, (s6 - s5 - 1.0))
        for k, (ms, en) in enumerate(items):
            tk = s5 + 0.8 + span * k / 5
            p = ease_out((t - tk) / 0.35)
            y = 330 + k * 105
            rect(cv, (990, y + 10, 1050, y + 70), YEL, a * p)
            paste(cv, text_img("B", "black", 44, BG), (1006, y + 12), a * p)
            paste(cv, text_img(ms, "black", 52, INK), (1080, y + 2), a * p)
            paste(cv, text_img(en, "med", 28, DIM), (1082, y + 66), a * p)
        q = ease_out((t - s6) / 0.5)
        paste(cv, text_img("Learning, disguised as play.", "black", 54, YEL), (990, 880), a * q)


def scene3(cv, c, t):
    chapter_card(cv, c, "02", "PROTECT", t)
    s0 = min(2.2, c.dur * 0.15)
    s1, s2, s3, s4 = c.cue("The Patient Safety"), c.cue("The Emergency"), c.cue("Infection control"), c.cue("And the medication")
    s1 = max(s1, s0 - 0.2)
    a = env(t, s0, s2 + 0.3)
    if a:
        kinetic(cv, "Underneath the fun: serious clinical work.", W / 2, 120, s0, t, 54, INK, "black", a, 0.04, 20)
    card(cv, "image25.jpg", 560, 470, 760, 560, s1, s2 + 0.3, t, drift=(0.3, 0.5))
    card(cv, "image6.jpg", 1340, 490, 720, 580, s1 + 0.5, s2 + 0.3, t, drift=(0.6, 0.4))
    a = env(t, s1, s2 + 0.3)
    if a:
        label(cv, "FALL RISK ASSESSMENT", (200, 790), s1 + 0.9, t, 46, alpha=a)
        subline(cv, "Identify mobility aids  →  diagnose risk factors  →  prevent the fall", (210, 880), s1 + 1.3, t, 30, INK, a)
    # CPR / Choking / AED kinetic list
    a = env(t, s2, s3 + 0.3)
    if a:
        label(cv, "EMERGENCY & TRAUMA UNIT · HANDS-ON", (160, 180), s2 + 0.1, t, 40, alpha=a)
        words = [("CPR", s2 + 0.6), ("CHOKING RESCUE", c.cue("choking")), ("AED", c.cue("defibrillator"))]
        for k, (w, tk) in enumerate(words):
            p = ease_out((t - tk) / 0.4)
            y = 330 + k * 200
            paste(cv, text_img(f"0{k+1}", "mono", 70, YEL), (170, y + 40), a * p)
            paste(cv, text_img(w, "black", 160, INK), (330 - 60 * (1 - p), y), a * p)
        q = ease_out((t - c.cue("into visitors")) / 0.5)
        paste(cv, text_img("…in visitors’ own hands.", "semi", 40, YEL), (340, 940), a * q)
    # Hand hygiene
    card(cv, "image4.jpg", 640, 540, 760, 800, s3, s4 + 0.3, t, zoom=0.12)
    a = env(t, s3, s4 + 0.3)
    if a:
        label(cv, "INFECTION PREVENTION", (1120, 380), s3 + 0.2, t, 50, alpha=a)
        label(cv, "& CONTROL", (1120, 460), s3 + 0.4, t, 50, alpha=a)
        subline(cv, "Clean hands. Safe lives.", (1126, 570), s3 + 0.7, t, 42, INK, a)
    # MedSafety
    card(cv, "image11.png", 520, 540, 620, 900, s4, c.dur, t, drift=(0.5, 0.9))
    a = env(t, s4, c.dur, 0.45, 0.4)
    if a:
        label(cv, "#MedSafetyWeek MALAYSIA 2026", (930, 260), s4 + 0.2, t, 40, alpha=a)
        sq = c.cue("side effects")
        kinetic(cv, "“Kesan sampingan ubat —", 1360, 470, sq, t, 58, INK, "black", a, 0.05, 20)
        boxes = kinetic(cv, "jangan abai, laporkan!”", 1360, 560, sq + 0.3, t, 70, YEL, "black", a, 0.05, 20)
        q = ease_out((t - c.cue("Report them")) / 0.4)
        paste(cv, text_img("Side effects — don’t ignore them. Report them.", "semi", 32, INK), (960, 660), a * q)


PARTNERS = [  # (logo, short name, cue phrase)
    ("image10.png", "JKN Kedah · TOBaTS", "Kedah's Pharmacy"),
    ("image8.png", "APM / JPAM Gurun", "The Civil Defence"),
    ("image24.jpg", "Klinik Nurul 24 Jam", "Klinik Nurul"),
    ("image19.png", "Diabetes Malaysia Yan", "The Diabetes"),
    ("image21.png", "Wall & Bone Physio", "Physiotherapists"),
    ("image1.jpg", "YaPEIM", "YaPEIM"),
    ("image5.png", "Tabung Haji Kedah", "Tabung Haji"),
]

def scene4(cv, c, t):
    chapter_card(cv, c, "03", "CONNECT", t)
    s0 = min(2.2, c.dur * 0.15)
    s_route = c.cue("And the reach")
    a = env(t, s0 - 0.1, s_route + 0.3)
    if a:
        cx, cy, R = W / 2, H / 2 - 10, 300
        kinetic(cv, "No single unit could build this alone.", W / 2, 100, s0 - 0.1, t, 50, INK, "black", a, 0.04, 20)
        n = len(PARTNERS)
        pts = []
        for k in range(n):
            ang = -math.pi / 2 + 2 * math.pi * k / n
            pts.append((cx + R * 1.45 * math.cos(ang), cy + R * math.sin(ang)))
        for k, (lg, nm, ph) in enumerate(PARTNERS):
            tk = c.cue(ph)
            line(cv, (cx, cy), pts[k], ease_io((t - tk) / 0.5), YEL, 4, a * 0.8)
        # hub
        p = ease_out((t - s0) / 0.5)
        hub = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        ImageDraw.Draw(hub).ellipse((10, 10, 290, 290), fill=YEL + (255,))
        paste(cv, hub, (cx - 150, cy - 150), a * p)
        for k, wtxt in enumerate(["HOSPITAL", "YAN"]):
            ti = text_img(wtxt, "black", 54, BG)
            paste(cv, ti, (cx - ti.width / 2, cy - 64 + k * 62), a * p)
        for k, (lg, nm, ph) in enumerate(PARTNERS):
            tk = c.cue(ph) + 0.3
            x, y = pts[k]
            logo_disc(cv, lg, x, y, 78, tk, t, a)
            ti = text_img(nm, "semi", 28, INK)
            q = ease_out((t - tk - 0.2) / 0.4)
            paste(cv, ti, (x - ti.width / 2, y + 92), a * q)
    # Cross-state route
    a = env(t, s_route, c.dur, 0.4, 0.4)
    if a:
        label(cv, "CROSS-STATE", (160, 160), s_route + 0.1, t, 44, alpha=a)
        p0, p1 = (420, 360), (980, 860)
        for (pt, nm, tk) in [(p0, "KEDAH · Hospital Yan", s_route + 0.3), (p1, "NEGERI SEMBILAN · CRC", s_route + 1.6)]:
            q = ease_out((t - tk) / 0.4)
            dot = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
            ImageDraw.Draw(dot).ellipse((6, 6, 54, 54), fill=YEL + (255,), outline=INK + (255,), width=4)
            paste(cv, dot, (pt[0] - 30, pt[1] - 30), a * q)
            paste(cv, text_img(nm, "black", 40, INK), (pt[0] + 50, pt[1] - 28), a * q)
        line(cv, p0, p1, ease_io((t - s_route - 0.5) / 1.2), YEL, 6, a, dash=18)
        card(cv, "image23.jpg", 1460, 520, 560, 700, s_route + 1.0, c.dur, t, drift=(0.5, 0.5))
        subline(cv, "Nursing Research Team, Clinical Research Centre", (1180, 900), s_route + 1.8, t, 28, INK, a)


def scene5(cv, c, t):
    chapter_card(cv, c, "04", "VISION", t)
    s0 = min(2.2, c.dur * 0.15)
    s1, s2, s3, s4 = c.cue("The carnival opens"), c.cue("It marks"), c.cue("Under the leadership"), c.cue("Hospital Yan Community Carnival, 2026")
    s1 = max(s1, s0 - 0.2)
    a = env(t, s0 - 0.2, s1 + 0.2)
    if a:
        kinetic(cv, "The timing is no accident.", W / 2, H / 2 + 220, s0 - 0.2, t, 72, INK, "black", a, 0.05)
    card(cv, "image28.png", 560, 540, 760, 820, s1, s3 + 0.3, t, drift=(0.5, 0.3))
    a = env(t, s1, s3 + 0.3)
    if a:
        label(cv, "18–24 NOV  ·  WAAW 2026", (1020, 300), s1 + 0.3, t, 50, alpha=a)
        subline(cv, "One Health, One Action: Prevent AMR Now", (1026, 400), s1 + 0.7, t, 36, INK, a)
        q = ease_out((t - s2) / 0.5)
        paste(cv, text_img("PATIENT SAFETY", "black", 100, INK), (1020, 520 + 30 * (1 - q)), a * q)
        paste(cv, text_img("DAY 2026", "black", 100, YEL), (1020, 630 + 30 * (1 - q)), a * q)
        subline(cv, "…celebrated where the public can see it.", (1026, 770), s2 + 0.6, t, 34, DIM, a)
    # Director card
    card(cv, "image15.jpg", W / 2, H / 2 - 60, 1600, 860, s3, s4 + 0.3, t, zoom=0.07, drift=(0.3, 0.5))
    a = env(t, s3, s4 + 0.3)
    if a:
        p = ease_out((t - s3 - 0.4) / 0.5)
        rect(cv, (160, 780, 160 + 1100 * p, 930), BG, a * 0.92)
        rect(cv, (160, 780, 172, 930), YEL, a * p)
        paste(cv, text_img("Dr. Tengku Nurshahril", "black", 64, INK), (200, 790), a * p)
        paste(cv, text_img("Hospital Director, Hospital Yan", "semi", 36, YEL), (204, 870), a * p)
        q = ease_out((t - c.cue("healthcare that walks")) / 0.5)
        if q:
            rect(cv, (160, 160, 160 + 1250 * q, 260), YEL, a)
            paste(cv, text_img("Healthcare that walks out to meet its community.", "black", 46, BG), (190, 185), a * q)
    # End title
    a = env(t, s4, c.dur + 1, 0.4, 0.01)
    if a:
        kinetic(cv, "HOSPITAL YAN", W / 2, H / 2 - 160, s4, t, 150, INK, alpha=a)
        kinetic(cv, "COMMUNITY CARNIVAL 2026", W / 2, H / 2, s4 + 0.25, t, 96, YEL, alpha=a)
        q = ease_io((t - s4 - 0.6) / 0.6)
        rect(cv, (W / 2 - 400 * q, H / 2 + 90, W / 2 + 400 * q, H / 2 + 96), INK, a)
        tg = c.cue("Together")
        ti = text_img("Together, for a safer Yan.", "semi", 52, INK)
        paste(cv, ti, (W / 2 - ti.width / 2, H / 2 + 130), a * ease_out((t - tg) / 0.6))
        ti = text_img("18 NOVEMBER 2026  ·  HOSPITAL YAN, KEDAH", "mono", 30, DIM)
        paste(cv, ti, (W / 2 - ti.width / 2, H / 2 + 230), a * ease_out((t - tg - 0.4) / 0.6))


SCENES = [scene1, scene2, scene3, scene4, scene5]


# ------------------------------------------------------------------- frame
def caption_chunks(text, maxlen=62):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > maxlen or (cur and cur[-1] in ".:!?—"):
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out

class Plan:
    def __init__(self, cfg, durations, speech, captions):
        self.cfg, self.durs, self.speech, self.captions = cfg, durations, speech, captions
        self.starts = np.cumsum([0] + durations[:-1]).tolist()
        self.total = sum(durations)
        self.caps = []
        for i, sc in enumerate(cfg["scenes"]):
            chunks = caption_chunks(sc["narration"])
            n = sum(len(c) for c in chunks)
            acc = 0
            for ch in chunks:
                a = LEAD + speech[i] * acc / n
                acc += len(ch) + 1
                b = LEAD + speech[i] * acc / n
                self.caps.append((self.starts[i] + a, self.starts[i] + b, ch))

    def frame(self, t):
        cv = backdrop().copy()
        i = max(k for k in range(5) if self.starts[k] <= t + 1e-9)
        sc = self.cfg["scenes"][i]
        lt = t - self.starts[i]
        SCENES[i](cv, Ctx(sc, self.speech[i], self.durs[i]), lt)
        # scene-boundary dip to black for a punchy cut
        if i > 0 and lt < 0.12:
            rect(cv, (0, 0, W, H), BG, 1 - lt / 0.12)
        # chrome: chapter chip, progress bar
        chip = text_img(f"{i+1:02d}  {sc['title'].upper()}", "mono", 22, DIM)
        paste(cv, chip, (60, 44), 0.9)
        rect(cv, (0, H - 10, W, H), (40, 41, 46))
        rect(cv, (0, H - 10, W * t / self.total, H), YEL)
        for s in self.starts[1:]:
            rect(cv, (W * s / self.total - 2, H - 16, W * s / self.total + 2, H), INK, 0.8)
        # captions
        if self.captions:
            for a, b, txt in self.caps:
                if a <= t < b:
                    ti = text_img(txt, "semi", 34, INK)
                    x, y = W / 2 - ti.width / 2, H - 92
                    rect(cv, (x - 18, y - 8, x + ti.width + 18, y + ti.height + 4), (0, 0, 0), 0.62)
                    paste(cv, ti, (x, y))
                    break
        cv.alpha_composite(vignette())
        cv.alpha_composite(grain_layers()[int(t * 12) % 6])
        return cv.convert("RGB")


# ------------------------------------------------------------------- main
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "hyan_carnival_2026.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview", action="store_true", help="960x540 @ 15fps, fast encode")
    ap.add_argument("--still", type=float, action="append", help="write a PNG at this time (s) and exit")
    ap.add_argument("--music", help="optional music bed, ducked under the narration")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "scenes.json")))
    durs, speech, audio = [], [], []
    for sc in cfg["scenes"]:
        p = os.path.join(ROOT, sc["audio"])
        if os.path.exists(p):
            d = probe(p); audio.append(p)
        else:
            d = sc["target_seconds"] - LEAD - TAIL; audio.append(None)
            print(f"[warn] {sc['audio']} missing — using {sc['target_seconds']}s of silence", file=sys.stderr)
        speech.append(d)
        durs.append(LEAD + d + TAIL)
    plan = Plan(cfg, durs, speech, not a.no_captions)
    print(f"[plan] scenes: {', '.join(f'{d:.1f}s' for d in durs)}  total {plan.total:.1f}s", file=sys.stderr)

    if a.still:
        os.makedirs(os.path.join(ROOT, "out", "stills"), exist_ok=True)
        for t in a.still:
            fn = os.path.join(ROOT, "out", "stills", f"t{t:06.2f}.png")
            plan.frame(t).save(fn); print(fn)
        return

    fps = 15 if a.preview else a.fps
    size = (960, 540) if a.preview else (W, H)
    nframes = int(math.ceil(plan.total * fps))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="hyan_")
    try:
        # 1) narration track: each scene = LEAD silence + voice + pad to scene length
        inputs, filt = [], []
        for i, (p, d) in enumerate(zip(audio, durs)):
            if p:
                inputs += ["-i", p]
                filt.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={int(LEAD*1000)}|{int(LEAD*1000)},apad,atrim=0:{d:.3f}[s{i}]")
            else:
                inputs += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                filt.append(f"[{i}:a]atrim=0:{d:.3f}[s{i}]")
        filt.append("".join(f"[s{i}]" for i in range(5)) + "concat=n=5:v=0:a=1[vo]")
        if a.music:
            inputs += ["-stream_loop", "-1", "-i", a.music]
            filt.append(f"[5:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.22,atrim=0:{plan.total:.3f},"
                        f"afade=t=out:st={plan.total-3:.3f}:d=3[mb];[mb][vo]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];"
                        f"[duck][vo]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5[aout]")
        else:
            filt.append("[vo]loudnorm=I=-16:TP=-1.5[aout]")
        wav = os.path.join(tmp, "mix.wav")
        if a.music:
            # vo is used twice; split it
            filt = [f.replace("concat=n=5:v=0:a=1[vo]", "concat=n=5:v=0:a=1,asplit=2[vo][vo2]") for f in filt]
            filt[-1] = filt[-1].replace("[duck][vo]amix", "[duck][vo2]amix")
        subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filt),
                        "-map", "[aout]", "-ar", "48000", wav], check=True)

        # 2) video: frames piped to x264, muxed with the mix
        enc = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "veryfast" if a.preview else "slow", "-crf", "28" if a.preview else "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", a.out]
        ff = subprocess.Popen(enc, stdin=subprocess.PIPE)
        with Pool(a.jobs, initializer=_init, initargs=(plan,)) as pool:
            for k, buf in enumerate(pool.imap(_render, ((f / fps, size) for f in range(nframes)), chunksize=4)):
                ff.stdin.write(buf)
                if k % (fps * 5) == 0:
                    print(f"\r[render] {k/fps:6.1f}s / {plan.total:.1f}s", end="", file=sys.stderr, flush=True)
        ff.stdin.close()
        if ff.wait() != 0:
            sys.exit("ffmpeg encode failed")
        print(f"\n[done] {a.out}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
