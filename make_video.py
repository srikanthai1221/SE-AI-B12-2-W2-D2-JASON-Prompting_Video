"""
Tamil Nadu 1000 Years Ago - illustrated animated video (c. 1000 CE, Chola era)
Pure Python (Pillow + NumPy) frames, piped to ffmpeg. 1280x720, 24 fps.
"""
import math, random, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FPS = 24
W, H = 3840, 2160            # world (background) resolution
OW, OH = 2560, 1440          # 2x render size (downsampled for anti-aliasing)
FW, FH = 1280, 720           # final size

# ----------------------------------------------------------------- utilities
def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))

def grad_rows(h, stops):
    """stops: list of (pos 0..1, (r,g,b)). returns h x 3 array"""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, 3))
    ps = [s[0] for s in stops]
    for c in range(3):
        out[:, c] = np.interp(ys, ps, [s[1][c] for s in stops])
    return out

def sky(img, y0, y1, stops):
    a = np.array(img)
    rows = grad_rows(y1 - y0, stops)
    a[y0:y1, :, :] = rows[:, None, :].astype(np.uint8)
    return Image.fromarray(a)

def glow(img, cx, cy, radius, color, strength=1.0):
    a = np.array(img).astype(np.float32)
    x0, x1 = max(0, int(cx - radius)), min(img.width, int(cx + radius))
    y0, y1 = max(0, int(cy - radius)), min(img.height, int(cy + radius))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    f = np.clip(1 - d, 0, 1) ** 2.2 * strength
    for c in range(3):
        a[y0:y1, x0:x1, c] += (color[c] - a[y0:y1, x0:x1, c] * 0.35) * f
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def ridge(d, y_base, amp, color, seed, step=60, w=W):
    r = random.Random(seed)
    pts = [(0, H)]
    y = y_base
    for x in range(0, w + step, step):
        y = y_base - amp * (0.5 + 0.5 * math.sin(x / 700 + seed)) - r.uniform(0, amp * 0.25)
        pts.append((x, y))
    pts.append((w, H))
    d.polygon(pts, fill=color)

# ----------------------------------------------------------------- drawing kit
def palm(d, x, y, h, trunk, leaf, lean=0.12, seed=0, sway=0.0):
    r = random.Random(seed)
    tw = max(2, h * 0.035)
    pts = []
    for i in range(11):
        t = i / 10
        pts.append((x + lean * h * t * t + sway * h * 0.03 * t * t, y - h * t))
    for i in range(10):
        d.line([pts[i], pts[i + 1]], fill=trunk, width=int(tw * (1 - 0.4 * i / 10)) + 1)
    tx, ty = pts[-1]
    for k in range(9):
        a = math.radians(-170 + k * 42 + r.uniform(-8, 8))
        L = h * r.uniform(0.32, 0.42)
        prev = (tx, ty)
        for s in range(1, 9):
            t = s / 8
            px = tx + math.cos(a) * L * t + sway * h * 0.02 * t
            py = ty + math.sin(a) * L * t + L * 0.55 * t * t
            d.line([prev, (px, py)], fill=leaf, width=max(2, int(h * 0.04 * (1 - t * 0.8))))
            prev = (px, py)
    d.ellipse([tx - h * 0.03, ty - h * 0.02, tx + h * 0.03, ty + h * 0.04], fill=trunk)

def person(d, x, y, h, skin, cloth, phase=0.0, pose="walk", facing=1, carry=None,
           female=False, upper=None, sil=None):
    if sil:
        skin = cloth = sil
        upper = sil
    hr = h * 0.075
    sw = math.sin(phase) * h * 0.11 if pose == "walk" else 0
    lw = max(2, int(h * 0.055))
    if pose == "bend":
        hip = (x, y - h * 0.47)
        sh = (x + facing * h * 0.26, y - h * 0.62)
        head = (x + facing * h * 0.38, y - h * 0.6)
        d.line([hip, (x - h * 0.06, y)], fill=skin, width=lw)
        d.line([hip, (x + h * 0.08, y)], fill=skin, width=lw)
        d.polygon([(hip[0] - h * 0.11, hip[1] - h * 0.04), (hip[0] + h * 0.1, hip[1] - h * 0.04),
                   (x + h * 0.12, y - h * 0.18), (x - h * 0.13, y - h * 0.18)], fill=cloth)
        d.line([hip, sh], fill=upper or skin, width=int(lw * 2.2))
        arm = math.sin(phase) * h * 0.06
        d.line([sh, (sh[0] + facing * h * 0.05 + arm, y - h * 0.12)], fill=skin, width=lw)
        d.ellipse([head[0] - hr, head[1] - hr, head[0] + hr, head[1] + hr], fill=skin)
        return
    if pose == "sit":
        hip = (x, y - h * 0.22)
        d.polygon([(x - h * 0.2, y), (x + h * 0.2, y), (x + h * 0.12, y - h * 0.25), (x - h * 0.12, y - h * 0.25)], fill=cloth)
        top = y - h * 0.58
        d.polygon([(x - h * 0.1, y - h * 0.22), (x + h * 0.1, y - h * 0.22), (x + h * 0.08, top), (x - h * 0.08, top)], fill=upper or skin)
        d.ellipse([x - hr, top - 2 * hr, x + hr, top], fill=skin)
        d.line([(x + h * 0.07, top + h * 0.05), (x + h * 0.17 + math.sin(phase) * h * 0.03, y - h * 0.28)], fill=skin, width=lw)
        return
    hip_y = y - h * 0.47
    sh_y = y - h * 0.8
    if female:
        d.polygon([(x - h * 0.08, sh_y), (x + h * 0.08, sh_y), (x + h * 0.15 + sw * 0.2, y - h * 0.02),
                   (x - h * 0.15 + sw * 0.2, y - h * 0.02)], fill=cloth)
        d.line([(x - h * 0.07, sh_y + h * 0.02), (x + h * 0.11, hip_y)], fill=mix(cloth, (255, 220, 120), 0.5), width=max(1, lw // 2))
    else:
        d.line([(x, hip_y), (x + sw, y)], fill=skin, width=lw)
        d.line([(x, hip_y), (x - sw, y)], fill=skin, width=lw)
        d.polygon([(x - h * 0.1, hip_y - h * 0.05), (x + h * 0.1, hip_y - h * 0.05),
                   (x + h * 0.12 + sw * 0.3, y - h * 0.12), (x - h * 0.12 + sw * 0.3, y - h * 0.12)], fill=cloth)
        d.polygon([(x - h * 0.09, sh_y), (x + h * 0.09, sh_y), (x + h * 0.075, hip_y - h * 0.04), (x - h * 0.075, hip_y - h * 0.04)], fill=upper or skin)
    # arms
    arm = -sw * 0.8
    if carry in ("pot", "basket"):
        d.line([(x + facing * h * 0.07, sh_y), (x + facing * h * 0.08, y - h * 1.02)], fill=skin, width=lw)
        d.line([(x - facing * h * 0.07, sh_y), (x - facing * h * 0.07 + arm, hip_y)], fill=skin, width=lw)
    else:
        d.line([(x + h * 0.08, sh_y), (x + h * 0.09 + arm, hip_y + h * 0.02)], fill=skin, width=lw)
        d.line([(x - h * 0.08, sh_y), (x - h * 0.09 - arm, hip_y + h * 0.02)], fill=skin, width=lw)
    hc = (x, sh_y - hr * 1.2)
    d.ellipse([hc[0] - hr, hc[1] - hr, hc[0] + hr, hc[1] + hr], fill=skin)
    if female and not sil:
        bx0, bx1 = sorted([hc[0] - facing * hr * 1.6, hc[0] - facing * hr * 0.6])
        d.ellipse([bx0, hc[1] - hr * 0.4, bx1, hc[1] + hr * 0.6], fill=(30, 22, 20))
    if carry == "pot":
        c = (205, 160, 60) if not sil else sil
        d.ellipse([x - hr * 1.3, hc[1] - hr * 3.2, x + hr * 1.3, hc[1] - hr * 0.8], fill=c)
    elif carry == "basket":
        c = (170, 120, 60) if not sil else sil
        d.polygon([(x - hr * 2.2, hc[1] - hr * 1.0), (x + hr * 2.2, hc[1] - hr * 1.0),
                   (x + hr * 1.6, hc[1] - hr * 2.2), (x - hr * 1.6, hc[1] - hr * 2.2)], fill=c)
        if not sil:
            for i in range(3):
                d.ellipse([x - hr * 1.4 + i * hr, hc[1] - hr * 2.9, x - hr * 0.4 + i * hr, hc[1] - hr * 1.9],
                          fill=[(220, 70, 40), (240, 180, 40), (90, 150, 50)][i])
    elif carry == "spear":
        hx = x + h * 0.12
        d.line([(hx, y - h * 0.15), (hx, y - h * 1.35)], fill=(90, 65, 40) if not sil else sil, width=max(1, lw // 2))
        d.polygon([(hx - h * 0.03, y - h * 1.33), (hx + h * 0.03, y - h * 1.33), (hx, y - h * 1.48)], fill=(200, 200, 210) if not sil else sil)
        d.ellipse([x - h * 0.2, y - h * 0.75, x + h * 0.02, y - h * 0.47], fill=(150, 40, 35) if not sil else sil,
                  outline=(210, 170, 60) if not sil else sil, width=max(1, lw // 2))
    elif carry == "drum":
        c = (140, 60, 40) if not sil else sil
        d.rounded_rectangle([x - h * 0.17, hip_y - h * 0.12, x + h * 0.17, hip_y + h * 0.08], radius=h * 0.05, fill=c)
        if not sil:
            d.line([(x - h * 0.17, hip_y - h * 0.02), (x + h * 0.17, hip_y - h * 0.02)], fill=(220, 180, 80), width=max(1, lw // 2))
    elif carry == "horn":
        c = (210, 170, 70) if not sil else sil
        d.line([(x + h * 0.04, sh_y - hr), (x + h * 0.35, sh_y - h * 0.3)], fill=c, width=lw)
        d.ellipse([x + h * 0.3, sh_y - h * 0.38, x + h * 0.42, sh_y - h * 0.24], fill=c)
    elif carry == "flag":
        px = x + h * 0.1
        d.line([(px, y - h * 0.3), (px, y - h * 1.5)], fill=(90, 65, 40) if not sil else sil, width=max(1, lw // 2))
        fl = math.sin(phase * 1.7) * h * 0.04
        d.polygon([(px, y - h * 1.5), (px + h * 0.45, y - h * 1.38 + fl), (px, y - h * 1.22)],
                  fill=(225, 120, 30) if not sil else sil)
        if not sil:  # tiger emblem hint
            d.ellipse([px + h * 0.08, y - h * 1.42, px + h * 0.18, y - h * 1.32], fill=(120, 40, 20))
    elif carry == "sack":
        c = (215, 195, 150) if not sil else sil
        d.ellipse([x - h * 0.2, sh_y - h * 0.18, x + h * 0.12, sh_y + h * 0.06], fill=c)
    elif carry == "lamp":
        hx, hy = x + h * 0.14, hip_y - h * 0.08
        d.polygon([(hx - h * 0.05, hy), (hx + h * 0.05, hy), (hx, hy + h * 0.04)], fill=(210, 170, 60))

def house(d, x, y, w, h, wall, roof, seed=0, stripes=False):
    d.rectangle([x, y - h, x + w, y], fill=wall)
    d.rectangle([x, y - h * 0.18, x + w, y], fill=mix(wall, (90, 60, 40), 0.35))   # thinnai platform
    if stripes:
        for i in range(int(w // (h * 0.12))):
            if i % 2 == 0:
                d.rectangle([x + i * h * 0.12, y - h * 0.95, x + (i + 1) * h * 0.12, y - h * 0.82], fill=(190, 60, 45))
    d.rectangle([x + w * 0.42, y - h * 0.7, x + w * 0.58, y - h * 0.18], fill=(60, 38, 28))
    for px in (x + w * 0.08, x + w * 0.92):
        d.rectangle([px - h * 0.03, y - h * 0.75, px + h * 0.03, y - h * 0.18], fill=mix(wall, (80, 50, 30), 0.5))
    rh = h * 0.55
    d.polygon([(x - w * 0.1, y - h * 0.95), (x + w * 1.1, y - h * 0.95), (x + w * 0.85, y - h - rh), (x + w * 0.15, y - h - rh)], fill=roof)
    for i in range(1, 5):
        yy = y - h * 0.95 - rh * i / 5
        f = i / 5
        d.line([(x - w * 0.1 + w * 0.25 * f, yy), (x + w * 1.1 - w * 0.25 * f, yy)], fill=mix(roof, (60, 25, 15), 0.4), width=max(1, int(h * 0.02)))

def vimana(d, cx, by, w, h, lit, shade, tiers=13, detail=True, sil=None):
    if sil:
        lit = shade = sil
    dark = mix(shade, (30, 20, 20), 0.35)
    def tier(x0, x1, y0, y1, col):
        d.rectangle([x0, y0, x1, y1], fill=col)
        if not sil:
            sx = x0 + (x1 - x0) * 0.62
            d.rectangle([sx, y0, x1, y1], fill=shade if col == lit else dark)
            d.line([(x0, y1), (x1, y1)], fill=dark, width=max(1, int((y1 - y0) * 0.12)))
            if detail and (x1 - x0) > 30:
                n = max(3, int((x1 - x0) / ((y1 - y0) * 1.1)))
                for i in range(n):
                    nx = x0 + (i + 0.5) * (x1 - x0) / n
                    nw = (x1 - x0) / n * 0.22
                    d.rectangle([nx - nw, y0 + (y1 - y0) * 0.3, nx + nw, y1 - (y1 - y0) * 0.15], fill=dark)
    # plinth & walls
    base_h = h * 0.1
    tier(cx - w / 2 - w * 0.04, cx + w / 2 + w * 0.04, by - base_h, by, lit)
    wall_h = h * 0.2
    y = by - base_h
    tier(cx - w / 2, cx + w / 2, y - wall_h, y, lit)
    if not sil and detail:
        for i in range(9):
            px = cx - w / 2 + (i + 0.5) * w / 9
            d.rectangle([px - w * 0.012, y - wall_h * 0.92, px + w * 0.012, y - wall_h * 0.08], fill=dark)
    y -= wall_h
    pyr_h = h * 0.53
    for k in range(tiers):
        f0, f1 = k / tiers, (k + 1) / tiers
        ww = w * (0.86 - 0.62 * f0)
        y0 = y - pyr_h * f1
        y1 = y - pyr_h * f0
        tier(cx - ww / 2, cx + ww / 2, y0, y1, lit)
    y -= pyr_h
    # griva + dome (shikhara)
    gw = w * 0.2
    d.rectangle([cx - gw / 2, y - h * 0.035, cx + gw / 2, y], fill=shade if not sil else sil)
    y -= h * 0.035
    dw = w * 0.3
    d.chord([cx - dw / 2, y - h * 0.12, cx + dw / 2, y + h * 0.06], 180, 360, fill=lit)
    if not sil:
        d.chord([cx + dw * 0.12, y - h * 0.12, cx + dw / 2, y + h * 0.06], 270, 360, fill=shade)
    y -= h * 0.03
    # kalasam
    kc = (215, 170, 70) if not sil else sil
    d.ellipse([cx - w * 0.025, y - h * 0.085, cx + w * 0.025, y - h * 0.04], fill=kc)
    d.polygon([(cx - w * 0.008, y - h * 0.08), (cx + w * 0.008, y - h * 0.08), (cx, y - h * 0.115)], fill=kc)

def gopuram(d, cx, by, w, h, lit, shade, sil=None):
    if sil:
        lit = shade = sil
    dark = mix(shade, (30, 20, 20), 0.35)
    d.rectangle([cx - w / 2, by - h * 0.28, cx + w / 2, by], fill=lit)
    if not sil:
        d.rectangle([cx + w * 0.12, by - h * 0.28, cx + w / 2, by], fill=shade)
        d.rectangle([cx - w * 0.1, by - h * 0.22, cx + w * 0.1, by], fill=(45, 30, 25))
    y = by - h * 0.28
    n = 7
    for k in range(n):
        f0, f1 = k / n, (k + 1) / n
        ww = w * (0.95 - 0.45 * f0)
        y0, y1 = y - h * 0.58 * f1, y - h * 0.58 * f0
        d.rectangle([cx - ww / 2, y0, cx + ww / 2, y1], fill=lit)
        if not sil:
            d.rectangle([cx + ww * 0.12, y0, cx + ww / 2, y1], fill=shade)
            d.line([(cx - ww / 2, y1), (cx + ww / 2, y1)], fill=dark, width=max(1, int(h * 0.006)))
            for i in range(5):
                nx = cx - ww / 2 + (i + 0.5) * ww / 5
                d.rectangle([nx - ww * 0.03, y0 + (y1 - y0) * 0.25, nx + ww * 0.03, y1 - (y1 - y0) * 0.15], fill=dark)
    y -= h * 0.58
    tw = w * 0.5
    d.rounded_rectangle([cx - tw / 2, y - h * 0.1, cx + tw / 2, y + 2], radius=h * 0.05, fill=lit)
    kc = (215, 170, 70) if not sil else sil
    for i in range(5):
        kx = cx - tw * 0.38 + i * tw * 0.19
        d.ellipse([kx - w * 0.02, y - h * 0.15, kx + w * 0.02, y - h * 0.09], fill=kc)

def elephant(d, x, y, s, phase, body=(95, 90, 95), cloth=(170, 35, 40), parasol=None, sil=None):
    if sil:
        body = cloth = sil
    leg = lambda px, ph: d.rectangle([px - s * 0.08 + math.sin(phase + ph) * s * 0.06, y - s * 0.55,
                                      px + s * 0.08 + math.sin(phase + ph) * s * 0.06, y], fill=mix(body, (0, 0, 0), 0.15) if not sil else sil)
    leg(x - s * 0.45, math.pi); leg(x + s * 0.3, 0)
    d.ellipse([x - s * 0.7, y - s * 1.05, x + s * 0.6, y - s * 0.4], fill=body)
    leg(x - s * 0.3, 0); leg(x + s * 0.45, math.pi)
    hx, hy = x + s * 0.62, y - s * 0.85
    d.ellipse([hx - s * 0.28, hy - s * 0.3, hx + s * 0.28, hy + s * 0.25], fill=body)
    tr = [(hx + s * 0.18, hy)]
    for i in range(1, 8):
        t = i / 7
        tr.append((hx + s * 0.25 + s * 0.12 * t + math.sin(phase * 0.5) * s * 0.03 * t, hy + s * 0.75 * t - s * 0.1 * t ** 4))
    d.line(tr, fill=body, width=int(s * 0.1))
    if not sil:
        d.line([(hx + s * 0.18, hy + s * 0.15), (hx + s * 0.38, hy + s * 0.28)], fill=(240, 235, 220), width=int(s * 0.04))
        d.ellipse([hx - s * 0.2, hy - s * 0.22, hx + s * 0.08, hy + s * 0.18], fill=mix(body, (0, 0, 0), 0.1))
        d.ellipse([hx + s * 0.08, hy - s * 0.1, hx + s * 0.14, hy - s * 0.04], fill=(20, 15, 15))
        # head ornament
        d.polygon([(hx - s * 0.05, hy - s * 0.3), (hx + s * 0.25, hy - s * 0.25), (hx + s * 0.18, hy + s * 0.05), (hx - s * 0.02, hy - s * 0.05)], fill=(200, 155, 50))
    # caparison
    d.polygon([(x - s * 0.5, y - s * 0.98), (x + s * 0.4, y - s * 0.98), (x + s * 0.45, y - s * 0.5), (x - s * 0.55, y - s * 0.5)], fill=cloth)
    if not sil:
        d.line([(x - s * 0.55, y - s * 0.52), (x + s * 0.45, y - s * 0.52)], fill=(225, 180, 60), width=int(s * 0.04))
        for i in range(6):
            px = x - s * 0.45 + i * s * 0.17
            d.ellipse([px - s * 0.03, y - s * 0.78, px + s * 0.03, y - s * 0.72], fill=(225, 180, 60))
    # howdah + parasol
    hc = (190, 140, 50) if not sil else sil
    d.rectangle([x - s * 0.35, y - s * 1.25, x + s * 0.25, y - s * 1.0], fill=hc)
    if parasol:
        pc = parasol if not sil else sil
        d.line([(x - s * 0.05, y - s * 1.25), (x - s * 0.05, y - s * 1.75)], fill=hc, width=max(2, int(s * 0.03)))
        d.chord([x - s * 0.45, y - s * 1.95, x + s * 0.35, y - s * 1.55], 180, 360, fill=pc)
        if not sil:
            for i in range(9):
                px = x - s * 0.42 + i * s * 0.095
                d.line([(px, y - s * 1.75), (px, y - s * 1.68)], fill=(225, 180, 60), width=max(1, int(s * 0.02)))
        # rider
        person(d, x - s * 0.05, y - s * 1.22, s * 0.38, (120, 75, 50), (240, 230, 210), pose="sit",
               upper=(200, 150, 50), sil=sil)

def bovine(d, x, y, s, phase, col=(235, 230, 220), horse=False, facing=1, sil=None, rider=None):
    if sil:
        col = sil
    lc = mix(col, (0, 0, 0), 0.2) if not sil else sil
    for px, ph in ((x - s * 0.35, math.pi), (x + s * 0.25, 0), (x - s * 0.25, 0), (x + s * 0.35, math.pi)):
        sw = math.sin(phase + ph) * s * 0.12
        d.line([(px, y - s * 0.5), (px + sw * facing, y)], fill=lc, width=max(2, int(s * 0.07)))
    d.ellipse([x - s * 0.55, y - s * 0.85, x + s * 0.5, y - s * 0.42], fill=col)
    nx = x + facing * s * 0.45
    if horse:
        d.polygon([(nx - facing * s * 0.1, y - s * 0.75), (nx + facing * s * 0.15, y - s * 1.15), (nx + facing * s * 0.3, y - s * 1.05), (nx + facing * s * 0.12, y - s * 0.6)], fill=col)
        d.polygon([(nx + facing * s * 0.12, y - s * 1.15), (nx + facing * s * 0.5, y - s * 0.95), (nx + facing * s * 0.45, y - s * 0.85), (nx + facing * s * 0.22, y - s * 0.95)], fill=col)
        d.line([(x - facing * s * 0.5, y - s * 0.75), (x - facing * s * 0.7, y - s * 0.45)], fill=lc, width=max(2, int(s * 0.08)))
    else:
        d.ellipse([x - facing * s * 0.05 - s * 0.15, y - s * 1.0, x - facing * s * 0.05 + s * 0.15, y - s * 0.7], fill=col)  # hump
        d.ellipse([nx - s * 0.12, y - s * 0.85, nx + s * 0.3 * facing if facing > 0 else nx + s * 0.12, y - s * 0.55], fill=col)
        hx = nx + facing * s * 0.2
        d.polygon([(hx, y - s * 0.85), (hx + facing * s * 0.25, y - s * 0.75), (hx + facing * s * 0.2, y - s * 0.55), (hx, y - s * 0.6)], fill=col)
        d.line([(hx, y - s * 0.85), (hx - facing * s * 0.05, y - s * 1.05)], fill=(90, 70, 50) if not sil else sil, width=max(2, int(s * 0.04)))
    if rider:
        person(d, x, y - s * 0.75, s * 0.75, (120, 75, 50), rider, pose="sit", upper=(170, 40, 40), sil=sil)

def ship(d, x, y, s, hull=(110, 70, 40), sail=(235, 220, 185), sil=None, flag=(200, 60, 40)):
    if sil:
        hull = sail = flag = sil
    d.polygon([(x - s, y - s * 0.22), (x + s * 1.05, y - s * 0.3), (x + s * 0.75, y + s * 0.1), (x - s * 0.7, y + s * 0.1)], fill=hull)
    if not sil:
        d.line([(x - s * 0.9, y - s * 0.12), (x + s * 0.95, y - s * 0.18)], fill=mix(hull, (0, 0, 0), 0.35), width=max(1, int(s * 0.03)))
    for mx, mh, sw in ((x - s * 0.35, s * 1.1, s * 0.45), (x + s * 0.35, s * 1.35, s * 0.55)):
        d.line([(mx, y - s * 0.2), (mx, y - s * 0.2 - mh)], fill=mix(hull, (0, 0, 0), 0.3) if not sil else sil, width=max(2, int(s * 0.04)))
        top = y - s * 0.2 - mh * 0.95
        bot = y - s * 0.32
        d.polygon([(mx - sw / 2, top), (mx + sw / 2, top), (mx + sw / 2 + s * 0.05, (top + bot) / 2), (mx + sw / 2, bot), (mx - sw / 2, bot), (mx - sw / 2 + s * 0.05, (top + bot) / 2)], fill=sail)
        if not sil:
            for i in range(1, 4):
                yy = top + (bot - top) * i / 4
                d.line([(mx - sw / 2 + s * 0.04, yy), (mx + sw / 2 + s * 0.04, yy)], fill=mix(sail, (120, 90, 60), 0.4), width=max(1, int(s * 0.015)))
    mx = x + s * 0.35
    d.polygon([(mx, y - s * 1.55), (mx + s * 0.2, y - s * 1.5), (mx, y - s * 1.45)], fill=flag)

def birds(d, cx, cy, s, n, t, col, seed=1):
    r = random.Random(seed)
    for i in range(n):
        ox, oy = r.uniform(-1, 1) * s * 6, r.uniform(-1, 1) * s * 2.5
        fl = math.sin(t * 9 + i * 1.3) * s * 0.35
        x, y = cx + ox, cy + oy
        d.line([(x - s, y - fl), (x, y), (x + s, y - fl)], fill=col, width=max(1, int(s * 0.18)))

# ----------------------------------------------------------------- camera
class Cam:
    def __init__(self, cx, cy, cw):
        self.cw = cw
        self.ch = cw * 9 / 16
        self.x0 = min(max(cx - self.cw / 2, 0), W - self.cw)
        self.y0 = min(max(cy - self.ch / 2, 0), H - self.ch)
        self.s = OW / self.cw
    def X(self, x): return (x - self.x0) * self.s
    def Y(self, y): return (y - self.y0) * self.s
    def S(self, v): return v * self.s
    def render(self, bg):
        return bg.resize((OW, OH), Image.BILINEAR, box=(self.x0, self.y0, self.x0 + self.cw, self.y0 + self.ch))

def lerp(a, b, t): return a + (b - a) * t

def finish(img2x):
    return img2x.reduce(2)

# ----------------------------------------------------------------- scenes
class Scene:
    dur = 11
    def build(self): pass
    def frame(self, t): raise NotImplementedError

# 1 ─ Aerial sunrise over the Kaveri delta
class Delta(Scene):
    dur = 11
    def build(self):
        hz = 900
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, hz + 40, [(0, (40, 44, 92)), (0.45, (150, 88, 112)), (0.8, (238, 150, 110)), (1, (255, 196, 130))])
        img = glow(img, 2350, 840, 1100, (255, 210, 150), 0.9)
        img = glow(img, 2350, 840, 260, (255, 245, 210), 1.0)
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse([2350 - 95, 840 - 95, 2350 + 95, 840 + 95], fill=(255, 238, 200))
        ridge(d, hz - 10, 70, (150, 100, 125), 3)
        ridge(d, hz + 5, 40, (125, 92, 112), 7)
        vimana(d, 1450, hz + 12, 120, 230, None, None, sil=(118, 86, 104), detail=False)
        vimana(d, 3150, hz + 14, 70, 130, None, None, sil=(122, 90, 108), detail=False)
        d.rectangle([0, hz, W, H], fill=(80, 105, 70))
        vp = 1900
        dep = lambda v: hz + (H - hz) * v ** 1.7
        sc = lambda v: 0.06 + 0.94 * v
        r = random.Random(11)
        greens = [(70, 128, 62), (92, 150, 70), (112, 162, 78), (64, 116, 58), (130, 170, 85), (84, 140, 60)]
        rows = 46
        for i in range(rows):
            v0, v1 = i / rows, (i + 1) / rows
            y0, y1 = dep(v0), dep(v1)
            s0, s1 = sc(v0), sc(v1)
            haze = 1 - v1 ** 0.6
            u = -vp / s0 - 400
            while u < (W - vp) / s0 + 400:
                fw = r.uniform(180, 420)
                c = r.choice(greens)
                if r.random() < 0.13:
                    c = (205, 165, 140)   # flooded field reflecting sunrise
                c = mix(c, (215, 160, 140), haze * 0.75)
                pts = [(vp + u * s0, y0), (vp + (u + fw) * s0, y0), (vp + (u + fw) * s1, y1), (vp + u * s1, y1)]
                d.polygon(pts, fill=c, outline=mix(c, (60, 80, 45), 0.35))
                u += fw
        # river
        riv = lambda v: -1300 * (1 - v) ** 0.9 * math.cos(v * 2.4) + 380 * math.sin(v * 9)
        N = 160
        for i in range(N):
            v0, v1 = i / N, (i + 1) / N
            hw0, hw1 = 300, 300
            col = mix((255, 200, 140), (200, 140, 150), v0)
            pts = [(vp + (riv(v0) - hw0) * sc(v0), dep(v0)), (vp + (riv(v0) + hw0) * sc(v0), dep(v0)),
                   (vp + (riv(v1) + hw1) * sc(v1), dep(v1)), (vp + (riv(v1) - hw1) * sc(v1), dep(v1))]
            d.polygon(pts, fill=col)
        # palms & hamlets
        items = []
        for _ in range(200):
            v = r.uniform(0.02, 1.0) ** 0.8
            side = r.choice([-1, 1])
            u = riv(v) + side * r.uniform(340, 900)
            if r.random() < 0.35:
                u = r.uniform(-vp / sc(v), (W - vp) / sc(v))
            items.append((v, u))
        items.sort()
        for v, u in items:
            x, y, s = vp + u * sc(v), dep(v), sc(v)
            dk = mix((190, 130, 125), (28, 38, 30), min(1, v * 1.6))
            if r.random() < 0.12:
                house(d, x, y, 260 * s, 140 * s, mix((215, 170, 140), dk, 0.4), mix((150, 80, 70), dk, 0.4))
            else:
                palm(d, x, y, 420 * s, dk, dk, lean=r.uniform(-0.15, 0.15), seed=int(u))
        for k in range(14):
            yy = hz + k * 14
            d.rectangle([0, yy - 30, W, yy + 30], fill=(255, 215, 190, 16))
        self.bg = img
    def frame(self, t):
        p = ease(t / self.dur)
        cam = Cam(lerp(1920, 1700, p), lerp(1080, 1250, p), lerp(3840, 2500, p))
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        bx = lerp(600, 2900, t / self.dur)
        birds(d, cam.X(bx), cam.Y(620 - t * 12), cam.S(22), 9, t, (40, 30, 45))
        return finish(img)

# 2 ─ Village, paddies and irrigation tank
class Village(Scene):
    dur = 11
    def build(self):
        hz = 960
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, hz + 20, [(0, (110, 160, 210)), (0.7, (200, 215, 220)), (1, (245, 228, 196))])
        img = glow(img, 600, 300, 1200, (255, 240, 200), 0.5)
        d = ImageDraw.Draw(img, "RGBA")
        ridge(d, hz - 40, 120, (150, 170, 178), 2)
        ridge(d, hz, 60, (128, 150, 138), 5)
        vimana(d, 2950, hz + 40, 260, 520, (205, 185, 165), (170, 155, 145), detail=True)
        d.rectangle([0, hz, W, H], fill=(96, 140, 72))
        # irrigation tank (eri)
        d.polygon([(0, hz + 20), (1500, hz + 20), (1650, 1190), (0, 1190)], fill=(128, 170, 188))
        for i in range(6):
            d.rectangle([0, 1190 + i * 16, 1700 - i * 6, 1206 + i * 16], fill=mix((150, 140, 125), (110, 100, 90), i / 6))
        r = random.Random(4)
        for i in range(30):
            palm(d, r.uniform(1500, 3840), hz + r.uniform(60, 160), r.uniform(330, 460), (95, 80, 60), (45, 105, 55), lean=r.uniform(-0.15, 0.15), seed=i)
        # village row
        x = 1650
        while x < 3900:
            w = r.uniform(170, 260)
            house(d, x, 1260, w, r.uniform(110, 150), r.choice([(228, 208, 172), (214, 168, 128), (235, 222, 196)]),
                  r.choice([(155, 72, 48), (172, 88, 52), (140, 64, 44)]), stripes=r.random() < 0.3)
            x += w + r.uniform(30, 90)
        for i in range(10):
            palm(d, r.uniform(1700, 3840), 1275, r.uniform(380, 520), (100, 80, 58), (38, 98, 50), lean=r.uniform(-0.15, 0.15), seed=100 + i)
        # paddy fields with perspective
        top = 1290
        vp = 1900
        rows = 14
        for i in range(rows):
            v0, v1 = i / rows, (i + 1) / rows
            y0, y1 = top + (H - top) * v0 ** 1.5, top + (H - top) * v1 ** 1.5
            s0, s1 = 0.3 + 0.7 * v0, 0.3 + 0.7 * v1
            u = -vp / s0 - 300
            while u < (W - vp) / s0 + 300:
                fw = r.uniform(500, 900)
                c = r.choice([(88, 150, 66), (104, 165, 72), (76, 136, 60), (122, 172, 80)])
                if r.random() < 0.15:
                    c = (150, 185, 190)
                d.polygon([(vp + u * s0, y0), (vp + (u + fw) * s0, y0), (vp + (u + fw) * s1, y1), (vp + u * s1, y1)], fill=c,
                          outline=(150, 135, 90), width=4)
                u += fw
        # bund path
        d.polygon([(0, 1500), (3840, 1420), (3840, 1460), (0, 1545)], fill=(160, 135, 92))
        self.bg = img
        self.farmers = [(r.uniform(400, 3500), r.uniform(1650, 2050), r.uniform(0, 6), r.choice([(240, 232, 215), (220, 120, 50), (235, 225, 200)]), r.choice([True, False]))
                        for _ in range(7)]
    def frame(self, t):
        p = ease(t / self.dur)
        cam = Cam(lerp(1350, 2450, p), lerp(1250, 1300, p), 2600)
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        # ripples on the tank
        for i in range(10):
            y = 1010 + i * 17
            x = (i * 260 + t * 60) % 1500
            d.line([(cam.X(x), cam.Y(y)), (cam.X(x + 120), cam.Y(y))], fill=(220, 235, 240, 140), width=max(1, int(cam.S(4))))
        # women walking on the bund with pots
        for k, (sx, col) in enumerate([(300, (180, 40, 60)), (520, (40, 110, 90)), (800, (210, 130, 30))]):
            x = sx + t * 110
            y = 1522 - x * (85 / 3840)
            person(d, cam.X(x), cam.Y(y), cam.S(150), (120, 76, 52), col, phase=x * 0.05, carry="pot", female=True)
        # ploughing bullocks
        bx = 2300 + t * 45
        for off in (0, 40):
            bovine(d, cam.X(bx + off), cam.Y(1880 + off * 0.5), cam.S(170), t * 4 + off)
        d.line([(cam.X(bx - 120), cam.Y(1790)), (cam.X(bx - 260), cam.Y(1900))], fill=(100, 70, 40), width=max(2, int(cam.S(8))))
        person(d, cam.X(bx - 300), cam.Y(1910), cam.S(200), (115, 72, 50), (238, 230, 214), phase=t * 4, facing=1)
        for (x, y, ph, col, fem) in self.farmers:
            person(d, cam.X(x), cam.Y(y), cam.S(190), (118, 74, 50), col, phase=t * 2 + ph, pose="bend",
                   facing=1 if ph > 3 else -1, upper=(190, 50, 60) if fem else None)
        birds(d, cam.X(1200 + t * 160), cam.Y(700), cam.S(16), 7, t, (250, 250, 245), seed=3)
        return finish(img)

# 3 ─ Bustling marketplace
class Market(Scene):
    dur = 11
    def build(self):
        hz = 1000
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, hz + 300, [(0, (130, 180, 222)), (1, (242, 230, 205))])
        d = ImageDraw.Draw(img, "RGBA")
        gopuram(d, 2050, 1260, 620, 1180, (212, 180, 135), (165, 132, 100))
        r = random.Random(9)
        x = -50
        while x < 3900:
            w = r.uniform(260, 380)
            hh = r.uniform(180, 260)
            if abs(x + w / 2 - 2050) > 420:
                house(d, x, 1300, w, hh, r.choice([(232, 210, 170), (222, 180, 130), (240, 228, 205)]),
                      r.choice([(158, 74, 50), (170, 90, 55)]), stripes=r.random() < 0.4)
            x += w + 20
        d.rectangle([0, 1300, W, H], fill=(196, 158, 108))
        for _ in range(900):
            px, py = r.uniform(0, W), r.uniform(1300, H)
            d.ellipse([px, py, px + 10, py + 5], fill=(170, 132, 88))
        # stalls
        x = 60
        awn = [((225, 120, 30), (245, 225, 180)), ((180, 40, 45), (240, 200, 120)), ((40, 110, 100), (235, 220, 190)), ((210, 160, 40), (170, 60, 40))]
        while x < 3800:
            w = 380
            a, b = r.choice(awn)
            yb = 1560
            for px in (x, x + w):
                d.rectangle([px - 8, yb - 330, px + 8, yb], fill=(110, 75, 45))
            for i in range(8):
                c = a if i % 2 == 0 else b
                d.polygon([(x - 30 + i * (w + 60) / 8, yb - 340), (x - 30 + (i + 1) * (w + 60) / 8, yb - 340),
                           (x - 30 + (i + 1) * (w + 60) / 8, yb - 270), (x - 30 + i * (w + 60) / 8, yb - 270)], fill=c)
            d.rectangle([x, yb - 110, x + w, yb], fill=(130, 90, 55))
            kind = r.choice(["pots", "spice", "cloth", "fruit"])
            if kind == "pots":
                for i in range(5):
                    for j in range(2 - (i % 2)):
                        cx = x + 40 + i * 72
                        d.ellipse([cx - 32, yb - 170 - j * 55, cx + 32, yb - 110 - j * 55], fill=r.choice([(176, 92, 52), (160, 80, 45), (190, 110, 60)]), outline=(120, 60, 35), width=3)
            elif kind == "spice":
                for i in range(5):
                    cx = x + 45 + i * 72
                    d.polygon([(cx - 34, yb - 110), (cx + 34, yb - 110), (cx, yb - 175)], fill=r.choice([(200, 50, 30), (235, 175, 30), (150, 85, 40), (220, 110, 30), (110, 120, 40)]))
            elif kind == "cloth":
                for i in range(6):
                    c = r.choice([(180, 30, 60), (230, 150, 30), (40, 90, 150), (30, 120, 80), (240, 230, 210), (120, 40, 110)])
                    d.rectangle([x + 20 + i * 58, yb - 200, x + 70 + i * 58, yb - 110], fill=c)
                    d.line([(x + 20 + i * 58, yb - 160), (x + 70 + i * 58, yb - 160)], fill=(225, 185, 70), width=5)
            else:
                for i in range(14):
                    cx, cy = x + 30 + (i % 7) * 50, yb - 135 - (i // 7) * 40
                    d.ellipse([cx - 22, cy - 18, cx + 22, cy + 18], fill=r.choice([(120, 80, 40), (230, 200, 60), (90, 140, 50)]))
            person(d, x + w / 2, yb - 105, 150, (115, 72, 50), (240, 230, 210), pose="sit", upper=r.choice([(190, 50, 60), None, (220, 140, 40)]))
            x += w + 70
        self.bg = img
        self.walkers = []
        for row, (y, h) in enumerate([(1700, 230), (2010, 300)]):
            for i in range(11):
                fem = r.random() < 0.45
                self.walkers.append(dict(x0=r.uniform(-400, 4200), y=y + r.uniform(-25, 25), h=h * r.uniform(0.92, 1.05),
                                         v=r.choice([-1, 1]) * r.uniform(55, 95), fem=fem,
                                         col=r.choice([(180, 30, 60), (230, 150, 30), (240, 232, 214), (40, 110, 90), (130, 40, 110), (210, 90, 40)]),
                                         carry=r.choice([None, None, "basket", "pot", "sack"]) if not fem else r.choice([None, "basket", "pot"])))
        self.walkers.sort(key=lambda w: w["y"])
    def frame(self, t):
        p = ease(t / self.dur)
        cam = Cam(lerp(1400, 2450, p), 1330, 2700)
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        # potter at the wheel
        px, py = 2700, 1900
        cx, cy = cam.X(px), cam.Y(py)
        d.ellipse([cx - cam.S(120), cy - cam.S(30), cx + cam.S(120), cy + cam.S(30)], fill=(110, 75, 45))
        for k in range(6):
            a = t * 6 + k * math.pi / 3
            d.line([(cx, cy), (cx + math.cos(a) * cam.S(115), cy + math.sin(a) * cam.S(26))], fill=(80, 55, 35), width=max(1, int(cam.S(6))))
        d.ellipse([cx - cam.S(45), cy - cam.S(110), cx + cam.S(45), cy - cam.S(20)], fill=(185, 105, 60))
        person(d, cam.X(px - 170), cam.Y(py + 20), cam.S(230), (112, 70, 48), (238, 228, 210), phase=t * 6, pose="sit")
        for k in range(5):
            qx = px + 220 + k * 85
            d.ellipse([cam.X(qx - 38), cam.Y(py - 70), cam.X(qx + 38), cam.Y(py + 10)], fill=(176, 96, 55), outline=(120, 60, 35), width=max(1, int(cam.S(4))))
        for wk in self.walkers:
            x = wk["x0"] + wk["v"] * t
            person(d, cam.X(x), cam.Y(wk["y"]), cam.S(wk["h"]), (116, 72, 50), wk["col"], phase=x * 0.035,
                   facing=1 if wk["v"] > 0 else -1, female=wk["fem"], carry=wk["carry"],
                   upper=(200, 160, 60) if not wk["fem"] and wk["carry"] is None else None)
        return finish(img)

# 4 ─ Monumental temple at golden hour (tilt-up reveal)
class Temple(Scene):
    dur = 12
    def build(self):
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, H, [(0, (60, 70, 125)), (0.5, (200, 120, 100)), (0.85, (250, 175, 95)), (1, (255, 200, 120))])
        img = glow(img, 600, 900, 1400, (255, 200, 120), 0.6)
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(6):
            d.polygon([(200 + i * 120, 0), (330 + i * 120, 0), (1700 + i * 420, H), (1450 + i * 420, H)], fill=(255, 220, 150, 14))
        vimana(d, 1920, 1900, 1250, 1740, (232, 168, 98), (150, 96, 72))
        # prakaram colonnade
        for side in (-1, 1):
            x0 = 0 if side < 0 else 2660
            x1 = 1180 if side < 0 else W
            d.rectangle([x0, 1640, x1, 1900], fill=(196, 140, 92))
            d.rectangle([x0, 1620, x1, 1660], fill=(160, 108, 76))
            for px in np.arange(x0 + 40, x1, 110):
                d.rectangle([px, 1660, px + 34, 1900], fill=(214, 160, 104))
                d.rectangle([px + 24, 1660, px + 34, 1900], fill=(160, 110, 80))
        # Nandi mandapam
        mx = 640
        d.rectangle([mx - 260, 1780, mx + 260, 1900], fill=(186, 130, 86))
        for px in (mx - 230, mx - 80, mx + 80, mx + 230):
            d.rectangle([px - 18, 1560, px + 18, 1780], fill=(214, 158, 104))
        d.polygon([(mx - 300, 1560), (mx + 300, 1560), (mx + 230, 1470), (mx - 230, 1470)], fill=(200, 140, 90))
        d.rectangle([0, 1900, W, H], fill=(172, 128, 92))
        for i in range(0, W, 160):
            d.line([(i, 1900), (i - 300, H)], fill=(150, 108, 78), width=4)
        for j in range(1950, H, 70):
            d.line([(0, j), (W, j)], fill=(150, 108, 78), width=3)
        # deepa stambha
        d.rectangle([3050, 1250, 3100, 1900], fill=(130, 100, 80))
        for k in range(8):
            d.rectangle([3020, 1300 + k * 70, 3130, 1312 + k * 70], fill=(160, 120, 85))
        palm(d, 120, 1660, 900, (60, 45, 40), (50, 60, 45), lean=0.2, seed=5)
        palm(d, 3720, 1660, 1000, (60, 45, 40), (50, 60, 45), lean=-0.2, seed=6)
        self.bg = img
    def frame(self, t):
        p = ease(t / self.dur)
        cw = lerp(1700, 3840, p)
        cam = Cam(1920, lerp(1640, 1080, p), cw)
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        lamps = [(x, 1895) for x in range(1360, 2500, 95)] + [(3060 + 10, 1300 + k * 70) for k in range(8)]
        for i, (x, y) in enumerate(lamps):
            f = 0.75 + 0.25 * math.sin(t * 11 + i * 2.1) * math.sin(t * 7 + i)
            r_ = cam.S(46) * f
            d.ellipse([cam.X(x) - r_, cam.Y(y - 20) - r_, cam.X(x) + r_, cam.Y(y - 20) + r_], fill=(255, 190, 90, 60))
            d.ellipse([cam.X(x) - r_ * 0.3, cam.Y(y - 20) - r_ * 0.5, cam.X(x) + r_ * 0.3, cam.Y(y - 20) + r_ * 0.2], fill=(255, 235, 170))
        for k in range(8):
            x = 900 + k * 330 + t * 25 * (1 if k % 2 else -1)
            person(d, cam.X(x), cam.Y(2060 + (k % 3) * 25), cam.S(250), (110, 70, 48),
                   [(240, 230, 210), (190, 40, 70), (225, 140, 30)][k % 3], phase=x * 0.04,
                   female=k % 2 == 0, carry="basket" if k % 3 == 0 else ("lamp" if k % 3 == 1 else None),
                   upper=(240, 230, 210) if k % 2 else None)
        birds(d, cam.X(2600 - t * 90), cam.Y(500), cam.S(18), 6, t, (40, 30, 45), seed=8)
        return finish(img)

# 5 ─ Royal Chola procession
class Procession(Scene):
    dur = 12
    def build(self):
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, 1400, [(0, (215, 150, 105)), (1, (250, 210, 155))])
        img = glow(img, 3300, 700, 1300, (255, 225, 160), 0.6)
        d = ImageDraw.Draw(img, "RGBA")
        gopuram(d, 3100, 1420, 640, 1250, (226, 170, 112), (168, 118, 84))
        vimana(d, 1100, 1300, 300, 560, (210, 165, 120), (170, 130, 100))
        r = random.Random(21)
        x = -40
        while x < 3900:
            w = r.uniform(260, 360)
            if abs(x + w / 2 - 3100) > 400:
                house(d, x, 1450, w, r.uniform(200, 260), r.choice([(236, 214, 176), (222, 182, 136)]), r.choice([(150, 70, 48), (172, 88, 54)]), stripes=r.random() < 0.5)
                # festival pennants
                for k in range(6):
                    px = x + k * w / 6
                    d.polygon([(px, 1150), (px + w / 6, 1150), (px + w / 12, 1200)], fill=r.choice([(225, 120, 30), (180, 40, 45), (240, 220, 160)]))
            x += w + 25
        d.rectangle([0, 1450, W, H], fill=(186, 146, 100))
        # spectators
        for i in range(70):
            px = r.uniform(0, W)
            person(d, px, 1560 + r.uniform(-20, 30), r.uniform(170, 200), (110, 70, 48),
                   r.choice([(180, 30, 60), (230, 150, 30), (240, 232, 214), (40, 110, 90)]), pose="walk",
                   phase=0, female=r.random() < 0.5)
        self.bg = img
        # procession lineup, leader first (largest x)
        units = (["drum"] * 3 + ["horn"] * 2 + ["flag"] + ["spear"] * 5 + ["horse"] * 2 + ["elephant", "king", "elephant"] + ["spear"] * 5 + ["flag"] + ["horse"] * 2)
        self.units, x = [], 2600
        for u in units:
            gap = {"elephant": 430, "king": 470, "horse": 300}.get(u, 140)
            x -= gap / 2
            self.units.append((u, x, r.uniform(0, 6)))
            x -= gap / 2
    def frame(self, t):
        p = ease(t / self.dur)
        cam = Cam(lerp(1500, 2550, p), 1420, 2900)
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        v = 135
        y = 1880
        for u, x0, ph in self.units:
            x = x0 + v * t
            X, Y = cam.X(x), cam.Y(y)
            phase = x * 0.03 + ph
            if u in ("elephant", "king"):
                elephant(d, X, Y, cam.S(330 if u == "king" else 290), phase * 0.6,
                         cloth=(170, 35, 40) if u == "king" else (40, 95, 110),
                         parasol=(245, 240, 230) if u == "king" else (225, 120, 30))
            elif u == "horse":
                bovine(d, X, Y, cam.S(250), phase, col=r_col(x0), horse=True, rider=(230, 220, 200))
            else:
                bob = abs(math.sin(phase)) * cam.S(6)
                person(d, X, Y - bob, cam.S(270), (112, 70, 48), (240, 230, 214) if u != "spear" else (170, 40, 40),
                       phase=phase, carry=u, upper=(200, 150, 50) if u in ("drum", "horn") else None)
        return finish(img)

def r_col(seed):
    return [(120, 80, 55), (235, 230, 220), (70, 50, 40)][int(seed) % 3]

# 6 ─ Coromandel Coast at sunset
class Coast(Scene):
    dur = 12
    def build(self):
        hz = 1050
        self.hz = hz
        img = Image.new("RGB", (W, H))
        img = sky(img, 0, hz, [(0, (58, 42, 92)), (0.5, (190, 88, 92)), (0.85, (250, 150, 80)), (1, (255, 190, 110))])
        img = glow(img, 2650, 980, 1300, (255, 180, 100), 0.8)
        img = glow(img, 2650, 980, 300, (255, 230, 170), 1.0)
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse([2650 - 115, 980 - 115, 2650 + 115, 980 + 115], fill=(255, 225, 160))
        a = np.array(img)
        a[hz:, :, :] = grad_rows(H - hz, [(0, (210, 120, 95)), (0.25, (120, 70, 95)), (1, (28, 30, 62))])[:, None, :].astype(np.uint8)
        img = Image.fromarray(a)
        d = ImageDraw.Draw(img, "RGBA")
        for k in range(10):
            d.line([(0, 980 - k * 60), (W, 960 - k * 70)], fill=(255, 210, 170, 10), width=30)
        # port on the left
        sil = (40, 26, 36)
        for i, (x, w, h) in enumerate([(0, 420, 260), (380, 360, 200), (700, 300, 240), (960, 260, 170)]):
            house(d, x, 1260, w, h, sil, sil)
        d.rectangle([0, 1250, 1350, 1330], fill=(34, 22, 30))
        d.rectangle([600, 1300, 1650, 1330], fill=(48, 32, 38))
        for px in range(640, 1650, 90):
            d.rectangle([px, 1300, px + 14, 1400], fill=(40, 26, 34))
        for i, x in enumerate([60, 260, 520, 1120, 1280]):
            palm(d, x, 1270, 620 + i * 40, sil, sil, lean=0.18 - i * 0.07, seed=40 + i)
        for k in range(5):
            x = 300 + k * 230
            person(d, x, 1255, 150, None, None, sil=sil, carry="sack" if k % 2 else None)
        self.bg = img
    def frame(self, t):
        p = ease(t / self.dur)
        cam = Cam(lerp(1500, 2050, p), lerp(1180, 1090, p), lerp(2700, 3700, p))
        img = cam.render(self.bg)
        d = ImageDraw.Draw(img, "RGBA")
        hz = self.hz
        # sun path on the water
        for k in range(40):
            yy = hz + 8 + k * 26
            wv = 40 + k * 9 + 25 * math.sin(t * 3 + k)
            d.line([(cam.X(2650 - wv), cam.Y(yy)), (cam.X(2650 + wv * 0.8), cam.Y(yy))], fill=(255, 205, 140, max(30, 180 - k * 4)), width=max(1, int(cam.S(6))))
        # waves
        for k in range(28):
            yy = hz + 40 + k ** 1.5 * 9
            off = (t * (20 + k * 6) + k * 377) % 500
            for x in np.arange(-500 + off, W, 500):
                d.arc([cam.X(x), cam.Y(yy - 8), cam.X(x + 160 + k * 6), cam.Y(yy + 8)], 200, 340, fill=(255, 190, 150, 70), width=max(1, int(cam.S(3))))
        # docked ships bobbing
        for i, x in enumerate([1000, 1500]):
            bob = math.sin(t * 1.6 + i) * 6
            ship(d, cam.X(x), cam.Y(1420 + bob), cam.S(230), sil=(36, 24, 32))
        # ships sailing out
        for i, (x0, y0, s0) in enumerate([(1700, 1300, 260), (2300, 1180, 170), (3000, 1110, 110)]):
            x = x0 + t * (70 - i * 15)
            y = y0 - t * (6 - i * 1.5)
            s = s0 * (1 - t * 0.012)
            bob = math.sin(t * 1.3 + i * 2) * 4
            ship(d, cam.X(x), cam.Y(y + bob), cam.S(s), sil=(44, 28, 38))
        birds(d, cam.X(1900 + t * 120), cam.Y(650), cam.S(18), 8, t, (40, 25, 40), seed=12)
        return finish(img)

# ----------------------------------------------------------------- title cards
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_SERIF_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_TAMIL = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"

def card_base():
    img = Image.new("RGB", (FW, FH))
    img = sky(img, 0, FH, [(0, (34, 20, 16)), (1, (70, 34, 22))])
    img = glow(img, FW / 2, FH / 2, 700, (120, 60, 30), 0.6)
    d = ImageDraw.Draw(img, "RGBA")
    gold = (205, 160, 80)
    d.rectangle([40, 40, FW - 40, FH - 40], outline=gold, width=2)
    d.rectangle([52, 52, FW - 52, FH - 52], outline=(205, 160, 80, 120), width=1)
    for cx, cy in ((40, 40), (FW - 40, 40), (40, FH - 40), (FW - 40, FH - 40)):
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=gold)
    vimana(d, FW / 2, FH - 70, 90, 120, None, None, sil=(110, 60, 36), detail=False)
    return img

def text_c(d, y, s, font, fill):
    w = d.textlength(s, font=font)
    d.text(((FW - w) / 2, y), s, font=font, fill=fill)

class Title(Scene):
    dur = 5
    def build(self):
        self.base = card_base()
    def frame(self, t):
        img = self.base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        a1 = int(255 * ease((t - 0.3) / 1.2))
        a2 = int(255 * ease((t - 1.2) / 1.2))
        a3 = int(255 * ease((t - 2.0) / 1.2))
        text_c(d, 170, "தமிழ்நாடு", ImageFont.truetype(FONT_TAMIL, 46, layout_engine=ImageFont.Layout.RAQM), (225, 185, 105, a1))
        text_c(d, 240, "TAMIL NADU", ImageFont.truetype(FONT_SERIF_B, 76), (245, 225, 185, a1))
        text_c(d, 335, "1000 Years Ago", ImageFont.truetype(FONT_SERIF, 46), (225, 185, 105, a2))
        text_c(d, 420, "A journey through the Chola era  ·  c. 1000 CE", ImageFont.truetype(FONT_SERIF, 24), (220, 200, 170, a3))
        return img

class EndCard(Scene):
    dur = 6
    def build(self):
        self.base = card_base()
    def frame(self, t):
        img = self.base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        a1 = int(255 * ease((t - 0.3) / 1.2))
        a2 = int(255 * ease((t - 1.3) / 1.2))
        text_c(d, 230, "The Chola legacy lives on —", ImageFont.truetype(FONT_SERIF, 40), (245, 225, 185, a1))
        text_c(d, 290, "in stone, in bronze, and on the sea routes they sailed.", ImageFont.truetype(FONT_SERIF, 30), (225, 185, 105, a1))
        text_c(d, 430, "An artistic reconstruction inspired by the Chola period of Tamil Nadu", ImageFont.truetype(FONT_SERIF, 18), (200, 180, 150, a2))
        return img

# ----------------------------------------------------------------- timeline & render
XF = 1.0
SCENES = [Title(), Delta(), Village(), Market(), Temple(), Procession(), Coast(), EndCard()]
starts, s = [], 0.0
for sc in SCENES:
    starts.append(s)
    s += sc.dur - XF
TOTAL = s + XF

yy, xx = np.mgrid[0:FH, 0:FW]
VIG = (1 - 0.38 * (((xx - FW / 2) / (FW / 2)) ** 2 + ((yy - FH / 2) / (FH / 2)) ** 2) ** 1.4).clip(0.45, 1)[:, :, None].astype(np.float32)
RNG = np.random.default_rng(3)

def grade(arr, card):
    a = arr.astype(np.float32)
    if not card:
        a = a * VIG
        a += RNG.normal(0, 3.2, (FH, FW, 1)).astype(np.float32)
    return a

def make_audio(path, total):
    sr = 44100
    n = int(total * sr)
    t = np.arange(n) / sr
    root = 146.83  # D3 - tanpura-like drone
    drone = np.zeros(n)
    for f, a in [(root / 2, 0.30), (root, 0.35), (root * 1.5, 0.22), (root * 2, 0.18), (root * 3, 0.06), (root * 4, 0.04)]:
        drone += a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.11 * t))
    pluck = 0.6 + 0.4 * np.abs(np.sin(np.pi * t / 1.6)) ** 6
    drone *= pluck * 0.22
    sig = drone
    # drums during the procession
    p0 = starts[5]
    p1 = p0 + SCENES[5].dur
    beat = 0.42
    bt = p0 + 0.3
    k = 0
    while bt < p1 - 0.4:
        i0 = int(bt * sr)
        L = int(0.35 * sr)
        tt = np.arange(L) / sr
        f = 95 * np.exp(-tt * 6) + 55
        hit = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 9) * (0.55 if k % 4 == 0 else 0.32)
        hit += np.random.default_rng(k).normal(0, 1, L) * np.exp(-tt * 40) * 0.08
        env = min(1, (bt - p0) / 1.5, (p1 - bt) / 1.5)
        sig[i0:i0 + L] += hit[: max(0, min(L, n - i0))] * env
        bt += beat if k % 4 != 3 else beat * 0.5
        k += 1
    # ocean wash at the coast
    c0 = starts[6]
    noise = np.random.default_rng(5).normal(0, 1, n)
    kern = np.ones(300) / 300
    sea = np.convolve(noise, kern, mode="same") * 6
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 4.5)
    fade = np.clip((t - c0) / 2, 0, 1) * np.clip((c0 + SCENES[6].dur + 3 - t) / 3, 0, 1)
    sig += sea * swell * fade * 0.18
    sig *= np.clip(t / 2.5, 0, 1) * np.clip((total - t) / 3, 0, 1)
    sig = sig / max(1e-6, np.abs(sig).max()) * 0.8
    st = np.stack([sig, np.roll(sig, 220)], axis=1)
    data = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())

def render(out, only=None):
    if only is not None:   # preview frames: list of (scene_index, t)
        for si, tt in only:
            sc = SCENES[si]
            if not hasattr(sc, "_built"):
                sc.build(); sc._built = True
            img = sc.frame(tt)
            Image.fromarray(np.clip(grade(np.array(img), si in (0, 7)), 0, 255).astype(np.uint8)).save(f"preview_{si}_{tt}.png")
        return
    make_audio("audio.wav", TOTAL)
    nframes = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}",
                           "-r", str(FPS), "-i", "-", "-i", "audio.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    built = set()
    for fi in range(nframes):
        T = fi / FPS
        acc, wsum = None, 0.0
        for si, sc in enumerate(SCENES):
            t0 = starts[si]
            if t0 <= T < t0 + sc.dur:
                if si not in built:
                    sc.build(); built.add(si)
                lt = T - t0
                w = min(1.0, lt / XF if si > 0 else 1.0, (sc.dur - lt) / XF if si < len(SCENES) - 1 else 1.0)
                fr = grade(np.array(sc.frame(lt)), si in (0, len(SCENES) - 1))
                acc = fr * w if acc is None else acc + fr * w
                wsum += w
        for si, sc in enumerate(SCENES):     # free finished scenes
            if si in built and T > starts[si] + sc.dur and hasattr(sc, "bg"):
                del sc.bg
        a = acc / max(wsum, 1e-6)
        a *= min(1, T / 1.0, (TOTAL - T) / 1.5)
        ff.stdin.write(np.clip(a, 0, 255).astype(np.uint8).tobytes())
        if fi % 120 == 0:
            print(f"frame {fi}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        render(None, only=[(0, 3.5), (1, 5.0), (2, 5.0), (3, 5.0), (4, 2.0), (4, 11.0), (5, 6.0), (6, 6.0), (7, 4.0)])
    else:
        render(sys.argv[1] if len(sys.argv) > 1 else "tamil_nadu_1000_years_ago.mp4")
