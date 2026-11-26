import sys, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import animate as A

FPS, OW, OH = A.FPS, A.OUT_W, A.OUT_H
SEG = 24.0                      # window segment, multiple of the 12 s loop
TOTAL = A.T + SEG
NF = int(TOTAL * FPS)

TEXT = """Ven a jugar.
Tú eres la pureza.

Mi modo lírico es lluvia de ideas en tu mente.
Soy la organizadora de tus delirios nocturnos.

Represento lo que odias,
un sucio deseo.
El dolor que sentiste es mío:
siéntete infectado.

Soy el anzuelo de todos tus miedos.
Las serpientes están en el jardín.

404."""
LINES = TEXT.split('\n')

MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
MONOB = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
font = ImageFont.truetype(MONO, 27)
font404 = ImageFont.truetype(MONOB, 40)
ftitle = ImageFont.truetype(MONO, 18)
CW = font.getlength('M')
LH = 38

WW, WH = 1000, 34 + 28 + LH * (len(LINES) - 1) + 56 + 28
WX, WY = (OW - WW) // 2, (OH - WH) // 2
PADX, PADY = 44, 34 + 26

C_TEXT = (150, 255, 210)
C_DIM = (90, 200, 230)
C_404 = (255, 50, 190)
C_BORDER = (60, 220, 255)

# ---- typing schedule: (line, char) -> time
RATE = 30.0
sched = []  # list of (t, line_idx, n_chars)
t = 0.9
for li, line in enumerate(LINES):
    if line == '404.':
        t += 0.9
    for k in range(1, len(line) + 1):
        t += 1 / RATE
        sched.append((t, li, k))
    t += 0.5 if line == '' else 0.22
TYPE_END = t
OPEN_END, GLITCH404, CLOSE0, CLOSE1 = 0.6, 20.4, 22.3, 22.9
assert TYPE_END < GLITCH404 - 1, TYPE_END
sched_t = np.array([s[0] for s in sched])

def ease(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def typed_state(s):
    k = np.searchsorted(sched_t, s, side='right')
    if k == 0:
        return 0, 0, False
    _, li, n = sched[k - 1]
    typing = s < TYPE_END
    # if paused at end of line, cursor goes to next line start
    if n == len(LINES[li]) and k < len(sched) and s > sched_t[k - 1] + 0.12:
        return sched[k][1], 0, typing
    return li, n, typing

def draw_window(s, i):
    """RGBA layer of the window at segment time s."""
    # open/close scale
    if s < OPEN_END:
        sx = ease(s / 0.3); sy = ease((s - 0.25) / 0.35) if s > 0.25 else 0.0
    elif s >= CLOSE0:
        u = (s - CLOSE0) / (CLOSE1 - CLOSE0)
        sy = 1 - ease(u / 0.6); sx = 1 - ease((u - 0.5) / 0.5)
    else:
        sx = sy = 1.0
    if sx <= 0:
        return None
    img = Image.new('RGBA', (WW, WH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, WW - 1, WH - 1], fill=(6, 10, 28, 225))
    # faint scanlines
    for y in range(0, WH, 3):
        d.line([(0, y), (WW, y)], fill=(0, 0, 0, 40))
    # title bar
    d.rectangle([0, 0, WW - 1, 34], fill=(10, 40, 70, 245))
    d.text((14, 7), 'jardin.exe  —  /dev/sueños', font=ftitle, fill=C_DIM)
    for k, c in enumerate([(255, 60, 190), (255, 200, 60), (60, 220, 255)]):
        x = WW - 30 - k * 26
        d.rectangle([x, 10, x + 14, 24], fill=c)
    d.rectangle([0, 0, WW - 1, WH - 1], outline=C_BORDER, width=2)
    d.rectangle([5, 39, WW - 6, WH - 6], outline=(40, 120, 170), width=1)

    li, n, typing = typed_state(s) if s >= OPEN_END else (0, 0, True)
    glitch404 = GLITCH404 <= s < CLOSE0
    rng = np.random.default_rng(i)
    cur_xy = (PADX, PADY)
    for k, line in enumerate(LINES):
        if k > li:
            break
        txt = line if k < li else line[:n]
        y = PADY + k * LH
        if line == '404.':
            y += 6
            if glitch404 and txt:
                for c, off in (((255, 0, 80), -5), ((0, 220, 255), 5)):
                    d.text((PADX + off * rng.uniform(0.5, 2), y), txt, font=font404, fill=c + (170,))
                if rng.random() < 0.25:
                    txt = ''.join(rng.choice(list('4O#@%0')) if rng.random() < 0.5 else ch for ch in txt)
            d.text((PADX, y), txt, font=font404, fill=C_404)
            x_end = PADX + font404.getlength(txt)
        else:
            d.text((PADX, y), txt, font=font, fill=C_TEXT)
            x_end = PADX + font.getlength(txt)
        if k == li:
            cur_xy = (x_end + 3, y)
    # cursor: solid while typing, blinking otherwise
    blink_on = typing or int(s * 2.4) % 2 == 0
    if blink_on and s >= OPEN_END and not glitch404:
        cx, cy = cur_xy
        hgt = 44 if LINES[li] == '404.' else 30
        d.rectangle([cx, cy + 3, cx + CW * (1.2 if hgt > 40 else 1) - 1, cy + hgt], fill=C_TEXT)

    # glow
    glow = img.filter(ImageFilter.GaussianBlur(6))
    out = Image.alpha_composite(glow, img)
    if sx < 1 or sy < 1:
        w, h = max(2, int(WW * sx)), max(3, int(WH * sy))
        out = out.resize((w, h), Image.BILINEAR)
        if sy < 0.02:  # bright CRT line
            out = Image.new('RGBA', (w, 3), C_BORDER + (255,))
    return out

def compose(fi):
    if fi < A.N:
        return A.frame(fi)
    s = (fi - A.N) / FPS
    bg = A.frame(fi - A.N, glitch=False).astype(np.float32)
    # dim background while window is up
    if s < CLOSE1:
        dim = ease(s / 0.5) * (1 - ease((s - CLOSE0) / (CLOSE1 - CLOSE0)))
        bg *= 1 - 0.35 * dim
    frame = Image.fromarray(cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2RGB))
    win = draw_window(s, fi)
    if win is not None:
        x = WX + (WW - win.width) // 2
        y = WY + (WH - win.height) // 2
        if GLITCH404 <= s < CLOSE0 and fi % 5 == 0:
            x += np.random.default_rng(fi).integers(-14, 14)
        frame.paste(win, (x, y), win)
    out = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    # glitch bursts: window opening and the 404 sequence
    if s < 0.25 or (GLITCH404 <= s < CLOSE1 and (fi % 4 in (0, 1)) and np.random.default_rng(fi + 9).random() < 0.6):
        st = 0.45 if s < 0.25 else 0.8
        out = A.apply_glitch(out.astype(np.float32), s, st, fi).astype(np.uint8)
    return out

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        for sec in map(float, sys.argv[2:]):
            cv2.imwrite(f'fin_{sec:05.1f}.png', compose(int(sec * FPS)))
        print('type end', TYPE_END); sys.exit()
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                          '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                          'video/jardin_digital.mp4'], stdin=subprocess.PIPE)
    for fi in range(NF):
        p.stdin.write(compose(fi).tobytes())
        if fi % 150 == 0: print(fi, flush=True)
    p.stdin.close(); p.wait(); print('done')
