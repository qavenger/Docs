# -*- coding: utf-8 -*-
"""短期补偿参数拟合：在 ACES 2.0（+ sRGB 编码）上找一组全局曝光 / 对比度 / 饱和度，让画面尽量接近原来的 Filmic。
目标函数：OKLab 色差（×100）的平均值，样本 = 两张游戏截图反推的场景值 + ColorChecker（-1/0/+1 EV）。
输出 data/fit.json。"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import ue_tonemap as T, invert_filmic as IF, colorchecker as CC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

# 两张 Filmic 下的编辑器 PIE 截图（原图在 input/，括号里是裁掉编辑器界面与游戏 UI 后的区域）
SHOTS = [("dungeon", os.path.join(HERE, "input", "dungeon_filmic_screenshot.png"), (424, 178, 1864, 978)),
         ("selectrole", os.path.join(HERE, "input", "selectrole_filmic_screenshot.png"), (980, 345, 1560, 930))]


def load_scene(path, box, step=4):
    a = np.asarray(Image.open(path).convert("RGB").crop(box), dtype=np.float64) / 255.0
    s = a[::step, ::step].reshape(-1, 3)
    x, res = IF.invert_filmic(s)
    return x[res < 2]


def samples():
    xs = [load_scene(p, b) for _, p, b in SHOTS]
    R = CC.reflectance()
    cc = np.concatenate([R * 2.0 ** e for e in (-1, 0, 1)])
    return xs, cc


def lab(code):
    return T.oklab_from_display(code)


def delta_e(a, b):
    return np.linalg.norm(lab(a) - lab(b), axis=-1) * 100


def score(params, xs, cc, eotf):
    e, con, sat = params
    mult = T.ACES_SCENE_COLOR_MULTIPLIER * 2.0 ** e
    c = {"contrast": con, "saturation": sat}
    total = []
    for x in xs + [cc]:
        ref = T.pipeline(x, "filmic")
        out = T.pipeline(x, "aces2", eotf=eotf, mult=mult, cc=c)
        total.append(delta_e(out, ref).mean())
    return float(np.mean(total))


def golden(f, lo, hi, it=18):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
    return (a + b) / 2


def fit(xs, cc, eotf):
    p = [0.0, 1.0, 1.0]
    bounds = [(-1.0, 1.5), (0.8, 1.6), (0.8, 1.8)]
    for _ in range(4):
        for k in range(3):
            def f(v, k=k):
                q = list(p); q[k] = v
                return score(q, xs, cc, eotf)
            p[k] = golden(f, *bounds[k])
    return p, score(p, xs, cc, eotf)


def metrics(x, method, **kw):
    code = T.pipeline(x, method, **kw)
    L, C, h = T.oklch_from_display(code)
    return {"L_mean": float(L.mean()), "L_p05": float(np.percentile(L, 5)), "L_p95": float(np.percentile(L, 95)),
            "L_std": float(L.std()), "C_mean": float(C.mean())}


if __name__ == "__main__":
    xs, cc = samples()
    res = {"samples": {n: int(len(x)) for (n, _, _), x in zip(SHOTS, xs)}}
    res["aces_default"] = score([0, 1, 1], xs, cc, "bt1886")
    res["aces_srgb"] = score([0, 1, 1], xs, cc, "srgb")
    p, s = fit(xs, cc, "srgb")
    res["fit_srgb"] = {"ev": p[0], "contrast": p[1], "saturation": p[2], "dE": s}
    p2, s2 = fit(xs, cc, "bt1886")
    res["fit_bt1886"] = {"ev": p2[0], "contrast": p2[1], "saturation": p2[2], "dE": s2}
    # 取整成美术好填的数（0.05 档）后再评一次
    r = [round(p[0] * 20) / 20, round(p[1] * 20) / 20, round(p[2] * 20) / 20]
    res["fit_srgb_rounded"] = {"ev": r[0], "contrast": r[1], "saturation": r[2], "dE": score(r, xs, cc, "srgb")}
    r2 = [round(p2[0] * 20) / 20, round(p2[1] * 20) / 20, round(p2[2] * 20) / 20]
    res["fit_bt1886_rounded"] = {"ev": r2[0], "contrast": r2[1], "saturation": r2[2], "dE": score(r2, xs, cc, "bt1886")}
    # 逐场景指标
    per = {}
    for (n, _, _), x in zip(SHOTS, xs):
        mult_r = T.ACES_SCENE_COLOR_MULTIPLIER * 2.0 ** r[0]
        cc_r = {"contrast": r[1], "saturation": r[2]}
        per[n] = {
            "filmic": metrics(x, "filmic"),
            "aces_default": metrics(x, "aces2"),
            "aces_srgb": metrics(x, "aces2", eotf="srgb"),
            "aces_srgb_comp": metrics(x, "aces2", eotf="srgb", mult=mult_r, cc=cc_r),
            "dE_default": float(delta_e(T.pipeline(x, "aces2"), T.pipeline(x, "filmic")).mean()),
            "dE_srgb": float(delta_e(T.pipeline(x, "aces2", eotf="srgb"), T.pipeline(x, "filmic")).mean()),
            "dE_comp": float(delta_e(T.pipeline(x, "aces2", eotf="srgb", mult=mult_r, cc=cc_r), T.pipeline(x, "filmic")).mean()),
        }
    res["per_scene"] = per
    json.dump(res, open(os.path.join(OUT, "fit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
