# -*- coding: utf-8 -*-
"""生成报告用的全部图片（img/）与图表数据（data/report_data.json）。依赖 fit_compensation.py 先跑出 data/fit.json。
运行：python -I make_assets.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
import ue_tonemap as T, invert_filmic as IF, colorchecker as CC
from fit_compensation import SHOTS

ROOT = os.path.normpath(os.path.join(HERE, ".."))
IMG = os.path.join(ROOT, "img"); DATA = os.path.join(ROOT, "data")
os.makedirs(IMG, exist_ok=True)
fit = json.load(open(os.path.join(DATA, "fit.json"), encoding="utf-8"))
COMP = fit["fit_srgb_rounded"]
COMP_MULT = T.ACES_SCENE_COLOR_MULTIPLIER * 2.0 ** COMP["ev"]
COMP_CC = {"contrast": COMP["contrast"], "saturation": COMP["saturation"]}

VARIANTS = [
    ("filmic", "Filmic（UE 默认）", dict(method="filmic")),
    ("aces_default", "ACES 2.0 默认", dict(method="aces2")),
    ("aces_srgb", "ACES 2.0 + sRGB 编码", dict(method="aces2", eotf="srgb")),
    ("aces_comp", "ACES 2.0 + sRGB + 补偿", dict(method="aces2", eotf="srgb", mult=COMP_MULT, cc=COMP_CC)),
]
KW = {k: kw for k, _, kw in VARIANTS}


def run(x, kw):
    return T.pipeline(x, **kw)


def to8(code):
    return np.round(np.clip(code, 0, 1) * 255).astype(np.uint8)


out = {"comp": COMP, "comp_mult": COMP_MULT, "variants": [[k, n] for k, n, _ in VARIANTS]}

# ------------------------------------------------ 1. 灰阶曲线（像素值 vs 曝光档位）
ev = np.round(np.arange(-8, 10.0001, 0.1), 2)
xg = 0.18 * np.exp2(ev)
Xg = np.stack([xg, xg, xg], -1)
tone = {"ev": ev.tolist()}
codes = {}
for k, n, kw in VARIANTS:
    c = run(Xg, kw)[..., 1]
    codes[k] = c
    tone[k] = np.round(c * 255, 1).tolist()
out["tone"] = tone

def at(c, e):
    return float(np.interp(e, ev, c))

stats = {}
for k, n, kw in VARIANTS:
    c = codes[k]
    lum = lambda e: float(T.srgb_to_linear(np.array(at(c, e))))
    above = lambda thr: float(ev[np.argmax(c >= thr)]) if np.any(c >= thr) else None
    stats[k] = {
        "mid": round(at(c, 0) * 255, 1),
        "slope": round((at(c, 0.25) - at(c, -0.25)) / 0.5 * 255, 1),
        "lum_mid": round(lum(0) * 100, 2), "lum_m3": round(lum(-3) * 100, 3), "lum_m5": round(lum(-5) * 100, 3),
        "code_m2": round(at(c, -2) * 255, 1), "code_m4": round(at(c, -4) * 255, 1), "code_p2": round(at(c, 2) * 255, 1),
        "code_p4": round(at(c, 4) * 255, 1),
        "ev90": above(0.9), "ev99": above(0.99),
    }
out["tone_stats"] = stats

# ------------------------------------------------ 2. 饱和度随曝光（ColorChecker 18 个彩色块的平均 OKLCH 彩度）
R = CC.reflectance()[:18]
evs = np.round(np.arange(-3, 6.0001, 0.25), 2)
sat = {"ev": evs.tolist()}
for k, n, kw in VARIANTS:
    vals = []
    for e in evs:
        L, C, h = T.oklch_from_display(run(R * 2.0 ** e, kw))
        vals.append(round(float(C.mean()) * 100, 2))
    sat[k] = vals
out["sat"] = sat

# ------------------------------------------------ 3. 高饱和颜色随曝光的色相偏移
COLORS = [("火焰橙", (1.0, 0.30, 0.04)), ("魔法蓝", (0.08, 0.22, 1.0)), ("血红", (1.0, 0.04, 0.04)),
          ("奥术紫", (0.55, 0.08, 1.0)), ("毒液绿", (0.18, 1.0, 0.08)), ("冰青", (0.05, 0.8, 1.0)), ("金黄", (1.0, 0.75, 0.05))]
sweep_ev = np.round(np.arange(-2, 8.0001, 0.5), 2)
hue = {"names": [c[0] for c in COLORS], "filmic": [], "aces_srgb": []}
sweep_imgs = {k: [] for k, _, _ in VARIANTS}
for name, c in COLORS:
    c = np.array(c); c = c / (c @ T.sRGB_2_XYZ[1]) * 0.18
    ref_h = T.oklch_from_display(T.encode(c[None, :] * 0.5, "srgb"))[2][0]
    fine = np.arange(-2, 6.0001, 0.05)
    for k in ("filmic", "aces_srgb"):
        kw = KW[k]
        L, C, h = T.oklch_from_display(run(c[None, :] * np.exp2(fine)[:, None], kw))
        dh = ((h - ref_h + 180) % 360) - 180
        m = C > 0.04
        hue[k].append(round(float(np.abs(dh[m]).max()), 1))
    for k, n, kw in VARIANTS:
        sweep_imgs[k].append(to8(run(c[None, :] * np.exp2(sweep_ev)[:, None], kw)))
out["hue"] = hue
out["sweep_ev"] = sweep_ev.tolist()


def save_sweep(rows, path, cell=36, cell_h=48, gap=2):
    H = len(rows) * cell_h + (len(rows) - 1) * gap
    W = rows[0].shape[0] * cell
    canvas = np.full((H, W, 3), 0, np.uint8)
    for i, r in enumerate(rows):
        y = i * (cell_h + gap)
        for j, px in enumerate(r):
            canvas[y:y + cell_h, j * cell:(j + 1) * cell] = px
    Image.fromarray(canvas).save(path, optimize=True)


for k, _, _ in VARIANTS:
    save_sweep(sweep_imgs[k], os.path.join(IMG, "sweep_%s.png" % k))

# ------------------------------------------------ 4. 灰阶条
strip_ev = np.linspace(-8, 8, 1280)
xs = 0.18 * np.exp2(strip_ev)
for k, n, kw in VARIANTS:
    row = to8(run(np.stack([xs, xs, xs], -1), kw))
    Image.fromarray(np.repeat(row[None, :, :], 44, axis=0)).save(os.path.join(IMG, "strip_%s.png" % k), optimize=True)

# ------------------------------------------------ 5. ColorChecker
def save_checker(code, path, cell=56, gap=4):
    W = 6 * cell + 7 * gap; H = 4 * cell + 5 * gap
    canvas = np.full((H, W, 3), 24, np.uint8)
    px = to8(code)
    for i in range(24):
        r, cc_ = divmod(i, 6)
        y = gap + r * (cell + gap); x = gap + cc_ * (cell + gap)
        canvas[y:y + cell, x:x + cell] = px[i]
    Image.fromarray(canvas).save(path, optimize=True)


Rall = CC.reflectance()
checker = {}
for k, n, kw in VARIANTS:
    code = run(Rall, kw)
    save_checker(code, os.path.join(IMG, "checker_%s.png" % k))
    L, C, h = T.oklch_from_display(code)
    checker[k] = {"C_mean": round(float(C[:18].mean()) * 100, 2), "L_white": round(float(L[18]), 3), "L_black": round(float(L[23]), 3)}
out["checker"] = checker

# 色卡在 -1 / 0 / +1 档三种曝光下与 Filmic 的平均色差（与 fit_compensation.py 的样本一致）
Rset = np.concatenate([Rall * 2.0 ** e for e in (-1, 0, 1)])
ref_lab = T.oklab_from_display(run(Rset, VARIANTS[0][2]))
out["checker_dE"] = {k: round(float((np.linalg.norm(T.oklab_from_display(run(Rset, kw)) - ref_lab, axis=-1) * 100).mean()), 2)
                     for k, _, kw in VARIANTS}

# 要达到与 Filmic 相同的屏幕像素值，ACES 需要多给多少档曝光（灰阶）
match = []
for target in (20, 60, 118, 150, 180, 200, 220, 235, 245):
    row = {"code": target}
    for k, _, _ in VARIANTS:
        c = codes[k] * 255
        i = int(np.argmax(c >= target))
        row[k] = round(float(ev[i]), 1) if c[i] >= target else None
    match.append(row)
out["ev_match"] = match

# ------------------------------------------------ 6. 两张游戏截图的近似重映射
scenes = {}
for name, path, box in SHOTS:
    a = np.asarray(Image.open(path).convert("RGB").crop(box), dtype=np.float64) / 255.0
    x, res = IF.invert_filmic(a)
    info = {"size": [a.shape[1], a.shape[0]], "res_mean": round(float(res.mean()), 3),
            "res_gt2": round(float((res > 2).mean()) * 100, 3), "clip255": round(float((a.max(-1) >= 254.5 / 255).mean()) * 100, 3)}
    ref = run(x, VARIANTS[0][2])
    for k, n, kw in VARIANTS:
        code = run(x, kw)
        Image.fromarray(to8(code)).save(os.path.join(IMG, "scene_%s_%s.jpg" % (name, k)), quality=92, subsampling=0)
        L, C, h = T.oklch_from_display(code)
        dE = np.linalg.norm(T.oklab_from_display(code) - T.oklab_from_display(ref), axis=-1) * 100
        info[k] = {"L_p05": round(float(np.percentile(L, 5)), 3), "L_mean": round(float(L.mean()), 3),
                   "L_p95": round(float(np.percentile(L, 95)), 3), "L_std": round(float(L.std()), 3),
                   "C_mean": round(float(C.mean()) * 100, 2), "dE": round(float(dE.mean()), 2)}
    # 原图（截图本身）也存一份，用来核对 Filmic 正向复刻是否与原图一致
    Image.fromarray(to8(a)).save(os.path.join(IMG, "scene_%s_original.jpg" % name), quality=92, subsampling=0)
    scenes[name] = info
out["scenes"] = scenes

json.dump(out, open(os.path.join(DATA, "report_data.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(json.dumps({"tone_stats": stats, "checker": checker, "hue": hue, "scenes": scenes, "comp": COMP}, ensure_ascii=False, indent=1))
