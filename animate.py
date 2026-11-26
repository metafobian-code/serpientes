import sys, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

SRC = 'assets/src.png'
T, FPS = 12.0, 30
N = int(T * FPS)
OUT_W, OUT_H = 1920, 1080
TAU = 2 * math.pi
rng = np.random.default_rng(7)

orig = cv2.imread(SRC).astype(np.float32)  # BGR
H, W = orig.shape[:2]
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

# ---------------------------------------------------------------- panels (typing)
PANELS = [  # box, t_start, t_end
    ((28, 30, 246, 462), 0.2, 6.4),
    ((1455, 40, 1650, 420), 0.8, 5.8),
    ((36, 615, 242, 852), 4.2, 8.4),
]
BIN_BOX = (1462, 655, 1650, 762)
HG_BOX = (95, 498, 156, 592)

inpaint_mask = np.zeros((H, W), np.uint8)
gray = orig.max(axis=2)
panel_data = []
for (x0, y0, x1, y1), ts, te in PANELS:
    m = (gray[y0:y1, x0:x1] > 105).astype(np.uint8)
    inpaint_mask[y0:y1, x0:x1] |= m
    prof = cv2.GaussianBlur(m.sum(1).astype(np.float32).reshape(-1, 1), (1, 7), 0).ravel()
    cand = [y for y in range(2, len(prof) - 2) if prof[y] >= prof[y - 2:y + 3].max() and prof[y] > 6]
    peaks = []
    for y in sorted(cand, key=lambda y: -prof[y]):
        if all(abs(y - p) >= 15 for p in peaks):
            peaks.append(y)
    peaks.sort()
    lines = []
    for k, p in enumerate(peaks):
        ys = max(0, p - 10) if k == 0 else (peaks[k - 1] + p) // 2
        ye = min(len(prof), p + 10) if k == len(peaks) - 1 else (p + peaks[k + 1]) // 2
        xs = np.where(m[ys:ye].any(0))[0]
        if len(xs):
            lines.append([ys, ye, xs.min(), xs.max() + 1, p])
    # schedule proportional to width + pause
    widths = np.array([l[3] - l[2] for l in lines], float)
    dur = widths / 8.4 * 1.0 + 1.6  # in "char" units, pause per line
    total = dur.sum()
    t = ts
    for l, d, wd in zip(lines, dur, widths):
        d = d / total * (te - ts)
        l += [t, t + d * (wd / 8.4) / (wd / 8.4 + 1.6)]
        t += d
    erase = rng.uniform(10.5, 11.25, len(lines))
    panel_data.append(dict(box=(x0, y0, x1, y1), mask=m.astype(np.float32), lines=lines, erase=erase))

# binary + hourglass also removed from clean plate
bx0, by0, bx1, by1 = BIN_BOX
inpaint_mask[by0:by1, bx0:bx1] |= (gray[by0:by1, bx0:bx1] > 90).astype(np.uint8)
hx0, hy0, hx1, hy1 = HG_BOX
hg_alpha = np.clip((gray[hy0:hy1, hx0:hx1] - 60) / 80, 0, 1)
inpaint_mask[hy0:hy1, hx0:hx1] |= (gray[hy0:hy1, hx0:hx1] > 60).astype(np.uint8)
inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8), iterations=2)
clean = cv2.inpaint(orig.astype(np.uint8), inpaint_mask * 255, 4, cv2.INPAINT_TELEA).astype(np.float32)
for d in panel_data:
    x0, y0, x1, y1 = d['box']
    d['diff'] = orig[y0:y1, x0:x1] - clean[y0:y1, x0:x1]
    d['soft'] = cv2.GaussianBlur(cv2.dilate(d['mask'], np.ones((3, 3))), (3, 3), 0)

# base plate: panels/binary/hourglass removed; everything else original
base = orig.copy()
base[inpaint_mask > 0] = clean[inpaint_mask > 0]

hg_img = orig[hy0:hy1, hx0:hx1].copy()

# ---------------------------------------------------------------- binary font
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 20)
def bin_layer(seed_rows):
    w, h = bx1 - bx0, by1 - by0
    img = Image.new('L', (w * 2, h * 2), 0)
    dr = ImageDraw.Draw(img)
    for r, (a, b) in enumerate(seed_rows):
        y = (667 - by0 - 8 + r * 20) * 2
        for col, s in ((1470, a), (1556, b)):
            tmp = Image.new('L', (8 * 24, 30), 0)
            ImageDraw.Draw(tmp).text((0, 0), s, font=font, fill=255, spacing=0)
            tmp = tmp.crop((0, 2, 8 * 12, 26)).resize((80 * 2, 15 * 2), Image.LANCZOS)
            img.paste(tmp, ((col - bx0) * 2, y + 2), tmp)
    a = np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255
    return a
bin_rows = [[''.join(rng.choice(['0', '1'], 8)) for _ in range(2)] for _ in range(5)]
BIN_COLOR = np.array([235, 215, 90], np.float32)  # BGR cyan

# ---------------------------------------------------------------- globes
GLOBES = [((1550, 538), 79, 1.0), ((1076, 646), 79, -1.0)]
def globe_maps(c, r, theta):
    cx, cy = c
    ys, xs = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1].astype(np.float32)
    nx, ny = (xs - cx) / r, (ys - cy) / r
    d2 = nx * nx + ny * ny
    inside = d2 < 0.985
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    lat_c = np.sqrt(np.clip(1 - ny * ny, 1e-6, 1))
    lon = np.arctan2(nx, nz) + theta
    l = np.mod(lon + math.pi / 2, TAU)
    l = np.where(l > math.pi, TAU - l, l) - math.pi / 2
    l *= 0.85
    mx = np.where(inside, cx + r * lat_c * np.sin(l), xs).astype(np.float32)
    feather = np.clip((0.985 - d2) / 0.06, 0, 1).astype(np.float32)
    return mx, ys, feather

# ---------------------------------------------------------------- fire mask
hsv = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
fire = (((hh < 13) | (hh > 172)) & (ss > 170) & (vv > 150)).astype(np.float32)
fire[290:, 880:1250] = 0
fire[560:, :] = 0
fire[:, :255] = 0; fire[:, 1445:] = 0; 
fire = cv2.GaussianBlur(fire, (0, 0), 6)
fire = np.clip(fire * 1.6, 0, 1)

def wave(x, y, t, comps):
    s = 0
    for kx, ky, m, ph, a in comps:
        s = s + a * np.sin(kx * x + ky * y + TAU * m * t / T + ph)
    return s
FIRE_D = [(0.031, 0.047, -7, 0.3, 1.0), (0.013, 0.021, -5, 1.7, 0.7), (0.057, 0.09, -11, 2.2, 0.4)]
FIRE_B = [(0.011, 0.025, -6, 0.9, 1.0), (0.023, 0.04, -9, 2.9, 0.6)]

# ---------------------------------------------------------------- screens glow
b, g, r_ = orig[..., 0], orig[..., 1], orig[..., 2]
screen = ((b > 190) & (b - r_ > 60) & (orig.max(2) > 190)).astype(np.float32)
screen[:, :255] = 0; screen[:, 1445:] = 0
for (cx, cy), rr, _ in GLOBES:
    cv2.circle(screen, (cx, cy), rr + 6, 0, -1)
screen = cv2.GaussianBlur(screen, (0, 0), 10)
screen = screen / (screen.max() + 1e-6)
GLOW = np.array([255, 170, 60], np.float32)

# ---------------------------------------------------------------- embers
NE = 90
emb = dict(x=rng.uniform(270, 1430, NE), y=rng.uniform(300, 920, NE), k=rng.integers(1, 4, NE),
           ph=rng.uniform(0, 1, NE), rise=rng.uniform(150, 320, NE), sw=rng.uniform(5, 25, NE),
           sz=rng.uniform(1.0, 2.6, NE))

# ---------------------------------------------------------------- birds
BIRDS = [(0.0, 95, 26, 1.0), (0.33, 140, 20, 1.3), (0.62, 70, 16, 0.9), (0.8, 165, 22, 1.1)]

# ---------------------------------------------------------------- glitch events
GLITCH = [(2.7, 0.13), (6.9, 0.10), (9.3, 0.17), (10.55, 0.75)]

def apply_glitch(img, t, strength, seed):
    g = np.random.default_rng(seed)
    out = img.copy()
    sh = int(4 + 10 * strength)
    out[..., 2] = np.roll(img[..., 2], sh, axis=1)
    out[..., 0] = np.roll(img[..., 0], -sh, axis=1)
    for _ in range(int(6 + 14 * strength)):
        y0 = g.integers(0, H - 10); hgt = g.integers(4, 40)
        out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], g.integers(-60, 60), axis=1)
    for _ in range(int(2 + 6 * strength)):
        x0, y0 = g.integers(0, W - 120), g.integers(0, H - 80)
        w_, h_ = g.integers(40, 160), g.integers(20, 90)
        blk = out[y0:y0 + h_, x0:x0 + w_]
        if blk.size:
            sm = cv2.resize(blk, (max(1, blk.shape[1] // 12), max(1, blk.shape[0] // 12)), interpolation=cv2.INTER_AREA)
            out[y0:y0 + h_, x0:x0 + w_] = cv2.resize(sm, (blk.shape[1], blk.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out

CURSOR = np.array([150, 255, 140], np.float32)

def frame(i, glitch=True):
    i = i % N
    t = i / FPS
    img = base.copy()

    # fire: upward warp + flicker
    dy = wave(XX, YY, t, FIRE_D) * 3.0
    dx = wave(YY, XX, t, FIRE_D[:2]) * 1.2
    warped = cv2.remap(orig, XX + dx * fire, YY + np.abs(dy) * fire, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flick = 1 + 0.22 * wave(XX, YY, t, FIRE_B) * fire
    img = img * (1 - fire[..., None]) + np.clip(warped * flick[..., None], 0, 255) * fire[..., None]

    # globes
    for (c, rr, dirn) in GLOBES:
        mx, my, fe = globe_maps(c, rr, dirn * TAU * t / T)
        patch = cv2.remap(orig, mx, my, cv2.INTER_LINEAR)
        cx, cy = c
        sl = (slice(cy - rr, cy + rr + 1), slice(cx - rr, cx + rr + 1))
        img[sl] = img[sl] * (1 - fe[..., None]) + patch * fe[..., None]

    # screens pulse
    pulse = 0.5 + 0.5 * np.sin(TAU * 2 * t / T * 3 + XX * 0.013 + YY * 0.007)
    img += screen[..., None] * GLOW * (0.15 + 0.55 * pulse[..., None] ** 3)

    # skull CRT flicker
    sx0, sy0, sx1, sy1 = 305, 408, 425, 500
    sk = img[sy0:sy1, sx0:sx1]
    f = 1.0 + 0.18 * math.sin(TAU * 17 * t / T) + 0.12 * math.sin(TAU * 41 * t / T)
    bar = ((np.arange(sy1 - sy0) - ((t * 60) % 140 - 20)) / 6.0)
    bar = np.exp(-bar ** 2)[:, None, None] * 50
    scan = (1 - 0.18 * (np.arange(sy1 - sy0) % 3 == 0))[:, None, None]
    jit = 3 if (i % 47) < 2 else 0
    img[sy0:sy1, sx0:sx1] = np.clip(np.roll(sk, jit, axis=1) * f * scan + bar * np.array([1, 0.5, 1.2]), 0, 255)

    # panels typing
    for d in panel_data:
        x0, y0, x1, y1 = d['box']
        rev = np.zeros((y1 - y0, x1 - x0), np.float32)
        cur = None
        for li, (ly0, ly1, lx0, lx1, pk, ta, tb) in enumerate(d['lines']):
            te = d['erase'][li]
            if t < ta or t >= te:
                if t >= te and cur is None:
                    pass
                if t < ta and cur is None and (li == 0 or t >= d['lines'][li - 1][6]):
                    cur = (lx0, pk - 7, pk + 7)
                continue
            if te - t < 0.18 and ((i * 7 + li) % 3 == 0):
                continue
            p = ease((t - ta) / max(tb - ta, 1e-3)) if t < tb else 1.0
            p = min(1.0, (t - ta) / max(tb - ta, 1e-3))
            xr = lx0 + p * (lx1 - lx0)
            xr = lx0 + math.floor((xr - lx0) / 8.4) * 8.4 if p < 1 else lx1
            rev[ly0:ly1, :int(xr)] = 1
            if p < 1:
                cur = (int(xr), pk - 7, pk + 7)
        reg = img[y0:y1, x0:x1]
        reg += d['diff'] * (rev * d['soft'])[..., None]
        if t < 10.5 and cur is not None and int(t * 4) % 2 == 0:
            cx_, ly0, ly1 = cur
            reg[ly0:ly1, cx_ + 1:cx_ + 8] = reg[ly0:ly1, cx_ + 1:cx_ + 8] * 0.2 + CURSOR * 0.8

    # binary
    step = i // 3
    g = np.random.default_rng(1000 + step)
    rows = [[''.join(c if g.random() > 0.35 else g.choice(['0', '1']) for c in s) for s in row] for row in bin_rows]
    a = bin_layer(rows)
    if 10.5 < t < 11.6:
        a *= 0.0 if (i % 2) else 0.5
    glow = cv2.GaussianBlur(a, (0, 0), 2.5)
    reg = img[by0:by1, bx0:bx1]
    img[by0:by1, bx0:bx1] = reg * (1 - a[..., None]) + BIN_COLOR * a[..., None] + glow[..., None] * BIN_COLOR * 0.35

    # hourglass: flips 3 times per loop
    period = T / 3
    ph = (t % period) / period
    ang = 180 * ease((ph - 0.82) / 0.12)
    hcx, hcy = (hx1 - hx0) / 2, (hy1 - hy0) / 2
    M = cv2.getRotationMatrix2D((hcx, hcy), ang, 1.0)
    hgr = cv2.warpAffine(hg_img, M, (hx1 - hx0, hy1 - hy0))
    har = cv2.warpAffine(hg_alpha.astype(np.float32), M, (hx1 - hx0, hy1 - hy0))[..., None]
    reg = img[hy0:hy1, hx0:hx1]
    img[hy0:hy1, hx0:hx1] = reg * (1 - har) + hgr * har
    if ph < 0.8:  # sand stream
        for k in range(4):
            sy = hcy - 4 + ((t * 40 + k * 7) % 26)
            cv2.rectangle(img, (int(hx0 + hcx - 1), int(hy0 + sy)), (int(hx0 + hcx + 1), int(hy0 + sy + 2)), (230, 230, 120), -1)

    # embers
    layer = np.zeros((H, W), np.float32)
    lt = (t / T * emb['k'] + emb['ph']) % 1.0
    ex = emb['x'] + emb['sw'] * np.sin(TAU * (lt * 2 + emb['ph']))
    ey = emb['y'] - lt * emb['rise']
    al = np.sin(np.pi * lt) ** 1.5
    for x_, y_, a_, s_ in zip(ex, ey, al, emb['sz']):
        cv2.circle(layer, (int(x_), int(y_)), int(round(s_)), float(a_), -1, lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.6) * 2.2
    img += layer[..., None] * np.array([60, 160, 255], np.float32)

    # birds (clipped to centre artwork)
    birds = np.zeros((H, W), np.uint8)
    for off, by, sz, sp in BIRDS:
        u = (t / T + off) % 1.0
        x = 250 + u * 1200
        y = by + 12 * math.sin(TAU * (u * 3 + off))
        fl = math.sin(TAU * (t * sp * 3 + off))
        wy = -sz * 0.5 * fl
        pts = np.array([[x - sz, y + wy], [x - sz * 0.45, y + wy * 0.3 - 3], [x, y + 2],
                        [x + sz * 0.45, y + wy * 0.3 - 3], [x + sz, y + wy]], np.int32)
        cv2.polylines(birds, [pts], False, 255, max(2, sz // 7), lineType=cv2.LINE_AA)
    birds[:, :250] = 0; birds[:, 1448:] = 0
    ba = (birds.astype(np.float32) / 255)[..., None]
    img = img * (1 - ba) + np.array([12, 4, 10], np.float32) * ba

    img = np.clip(img, 0, 255)

    # glitch events
    for gt, gd in (GLITCH if glitch else []):
        if gt <= t < gt + gd:
            s = 0.75 if gd > 0.5 else 0.6
            if gd > 0.5 and (i % 2 == 1):
                continue
            img = apply_glitch(img, t, s, i)

    # camera: slow breathing zoom with drift
    z = 1.0 + 0.06 * (1 - math.cos(TAU * t / T)) / 2
    cw, ch = W / z, H / z
    cx = W / 2 + (W - cw) / 2 * 0.6 * math.sin(TAU * t / T)
    cy = H / 2 + (H - ch) / 2 * 0.5 * math.sin(TAU * 2 * t / T)
    M = np.array([[OUT_W / cw, 0, -(cx - cw / 2) * OUT_W / cw],
                  [0, OUT_H / ch, -(cy - ch / 2) * OUT_H / ch]], np.float32)
    out = cv2.warpAffine(img.astype(np.float32), M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        for i in map(int, sys.argv[2:]):
            cv2.imwrite(f'prev_{i:03d}.png', frame(i))
        cv2.imwrite('fire_mask.png', (fire * 255).astype(np.uint8))
        cv2.imwrite('screen_mask.png', (screen * 255).astype(np.uint8))
        sys.exit()
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                          '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                          'video/jardin_digital_loop.mp4'], stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(frame(i).tobytes())
        if i % 60 == 0: print(i, flush=True)
    p.stdin.close(); p.wait()
    print('done')
