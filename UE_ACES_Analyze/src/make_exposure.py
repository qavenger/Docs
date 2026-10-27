# -*- coding: utf-8 -*-
"""长期方案里「灯光 / 曝光 / Albedo」图解用到的数字和图片：data/exposure_data.json 与 img/expo_*.png、img/fx_*.png。
和 make_assets.py 用同一套 Filmic / ACES 2.0 复刻（ue_tonemap.py），build_html.py 读结果填进页面。

本工程的前提（UE 5.8.4 源码核对，见报告附录 A）：
  r.EyeAdaptationQuality=0 → 曝光锁定 ×1，PPV 的 Exposure Compensation 不参与计算。
  所以「正确曝光」= 18% 灰卡在主光下的场景值正好 0.18；此时任何正对主光的漫反射表面，场景值 ≈ 它的 BaseColor。
运行：python -I make_exposure.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
import ue_tonemap as T, colorchecker as CC

ROOT = os.path.normpath(os.path.join(HERE, ".."))
IMG = os.path.join(ROOT, "img"); DATA = os.path.join(ROOT, "data")

FILMIC = dict(method="filmic")
ACES = dict(method="aces2", eotf="srgb")                 # 长期标准：ACES 2.0 + sRGB 编码（UE 默认 ×1.5）
ACES_OCIO = dict(method="aces2", eotf="srgb", mult=1.0)  # DCC 里 OCIO 的 ACES 2.0（不乘 1.5）


def code(x, kw):
    """中性灰场景值 → 屏幕像素值（0–255，浮点）"""
    x = np.asarray(x, dtype=np.float64)
    X = np.stack([x, x, x], -1)
    return T.pipeline(X, **kw)[..., 1] * 255


def s8(lin):
    """线性值 → sRGB 8 bit（四舍五入）"""
    return int(np.round(float(T.linear_to_srgb(np.array(float(lin)))) * 255))


def hx(rgb8):
    return "#%02X%02X%02X" % tuple(rgb8)


def ev(lin):
    return float(np.log2(lin / 0.18))


def r1(v):
    return round(float(v), 1)


def to8(c):
    return np.round(np.clip(c, 0, 1) * 255).astype(np.uint8)


out = {}

# ------------------------------------------------ 1. 数值对照：UE 线性值 ↔ PS sRGB 值
# 刻度位置按 sRGB 值排布（和眼睛的感觉一致），上排标 PS 的 sRGB 值，下排标同一位置的 UE 线性值；major = 窄屏也显示
out["ruler"] = [{"s": c, "lin": round(float(T.srgb_to_linear(np.array(c / 255.0))), 3), "major": c in (0, 50, 118, 188, 240, 255)}
                for c in (0, 30, 50, 89, 118, 137, 188, 225, 240, 255)]
out["ps118_naive"] = round(118 / 255, 3)                      # 把 PS 的 118 直接 ÷255 填进常量
out["ps118_naive_ev"] = round(ev(118 / 255), 1)
out["ps128_lin"] = round(float(T.srgb_to_linear(np.array(128 / 255))), 3)
out["lin05_s"] = s8(0.5)

# ------------------------------------------------ 2. UE 官方实测 BaseColor（《Physically Based Materials》文档）
NONMETAL = [("木炭", 0.02), ("新沥青", 0.02), ("旧沥青", 0.08), ("裸土", 0.13), ("绿草", 0.21),
            ("沙漠沙", 0.36), ("新混凝土", 0.51), ("海冰", 0.56), ("新雪", 0.81)]
METAL = [("铁", (0.560, 0.570, 0.580)), ("铬", (0.550, 0.556, 0.554)), ("钛", (0.542, 0.497, 0.449)),
         ("镍", (0.660, 0.609, 0.526)), ("铂", (0.672, 0.637, 0.585)), ("钴", (0.662, 0.655, 0.634)),
         ("铝", (0.913, 0.921, 0.925)), ("银", (0.972, 0.960, 0.915)), ("金", (1.000, 0.766, 0.336)),
         ("铜", (0.955, 0.637, 0.538))]


def grey_item(name, v, note=""):
    c = s8(v)
    return {"name": name, "lin": v, "s": c, "hex": hx((c, c, c)), "ev": round(ev(v), 2),
            "filmic": r1(code(v, FILMIC)), "aces": r1(code(v, ACES)), "note": note}


out["nonmetal"] = [grey_item(n, v) for n, v in NONMETAL]
out["metal"] = [{"name": n, "lin": list(v), "s": [s8(x) for x in v], "hex": hx([s8(x) for x in v])} for n, v in METAL]

# 标定道具与安全范围的边界
out["props"] = {
    "floor_tol": grey_item("宽松下限", round(float(T.srgb_to_linear(np.array(30 / 255))), 3)),
    "black": grey_item("黑卡", 0.032),
    "grey": grey_item("18% 灰卡", 0.18),
    "white": grey_item("白卡", 0.87),
    "max": grey_item("BaseColor 1.0", 1.0),
}

# ColorChecker：真实颜料的颜色上限参考（sRGB 8 bit 与线性值）
lin_cc = CC.reflectance()
out["checker_ref"] = [{"name": CC.NAMES[i], "s": [int(v) for v in CC.SRGB8[i]], "hex": hx([int(v) for v in CC.SRGB8[i]]),
                       "lin": [round(float(v), 3) for v in lin_cc[i]]} for i in range(24)]

# ------------------------------------------------ 3. 正确曝光下：你画的值 → 屏幕上看到的值
out["painted"] = [{"s": s8(v), "lin": v, "filmic": r1(code(v, FILMIC)), "aces": r1(code(v, ACES))} for v in (0.032, 0.18, 0.5, 0.87, 1.0)]

# ------------------------------------------------ 4. Pixel Inspector：Final Color（线性）↔ 屏幕像素值；灰卡读数容差
out["final_color"] = [{"lin": v, "s": s8(v)} for v in (0.0319, 0.05, 0.1, 0.156, 0.181, 0.3, 0.5, 0.75, 1.0)]
out["grey_tol"] = [{"scene": v, "ev": round(ev(v), 2), "filmic": r1(code(v, FILMIC)), "aces": r1(code(v, ACES)),
                    "final_lin": round(float(T.srgb_to_linear(np.array(code(v, ACES) / 255))), 3)} for v in (0.16, 0.18, 0.20)]
out["grey_final_aces"] = round(float(T.srgb_to_linear(np.array(code(0.18, ACES) / 255))), 3)
out["grey_final_filmic"] = round(float(T.srgb_to_linear(np.array(code(0.18, FILMIC) / 255))), 3)


def final_lin(scene, kw):
    """Pixel Inspector 的 Final Color 读数（线性值）"""
    return round(float(T.srgb_to_linear(np.array(code(scene, kw) / 255))), 3)


out["white_final"] = {"filmic": final_lin(0.87, FILMIC), "aces": final_lin(0.87, ACES)}

# 工程默认开了局部曝光（r.DefaultFeature.LocalExposure.HighlightContrastScale / ShadowContrastScale = 0.8）：
# 大块区域相对中灰的档数 ×0.8（CalculateLogLocalExposure），只影响 Final Color，不影响 Scene Color
LE = 0.8
out["local_exposure"] = [{"ev": e, "ev_le": round(e * LE, 2), "aces": r1(code(0.18 * 2 ** e, ACES)), "aces_le": r1(code(0.18 * 2 ** (e * LE), ACES))}
                         for e in (-5.0, -2.5, 2.27)]

# ------------------------------------------------ 5. 档位尺上的标记
D = json.load(open(os.path.join(DATA, "report_data.json"), encoding="utf-8"))
ts = D["tone_stats"]
out["marks"] = {"black": round(ev(0.032), 2), "floor_tol": round(ev(0.013), 2), "white": round(ev(0.87), 2), "one": round(ev(1.0), 2),
                "aces90": ts["aces_srgb"]["ev90"], "aces99": ts["aces_srgb"]["ev99"],
                "filmic90": ts["filmic"]["ev90"], "filmic99": ts["filmic"]["ev99"]}

# ------------------------------------------------ 6. 自发光：比中灰亮 N 档 → Emissive 数值（曝光 ×1）与屏幕值
out["emissive"] = [{"stop": n, "value": round(0.18 * 2 ** n, 2), "filmic": r1(code(0.18 * 2 ** n, FILMIC)),
                    "aces": r1(code(0.18 * 2 ** n, ACES))} for n in range(0, 9)]


def stop_for(target, kw):
    e = np.arange(-8, 12.0001, 0.01)
    c = code(0.18 * np.exp2(e), kw)
    i = int(np.argmax(c >= target))
    return round(float(e[i]), 1) if c[i] >= target else None


out["emissive_target"] = []
for tgt in (200, 230, 245):
    a, f = stop_for(tgt, ACES), stop_for(tgt, FILMIC)
    out["emissive_target"].append({"code": tgt, "filmic": f, "aces": a,
                                   "filmic_v": round(0.18 * 2 ** f, 2), "aces_v": round(0.18 * 2 ** a, 2)})

# ------------------------------------------------ 7. 每类场景的正常曝光 / 光比（建议起点）→ 屏幕值
SCENES = [
    ("主城白天（晴）", "太阳 + 天光", 0.0, -2.5, "阴影里只有天光"),
    ("阴天 / 雾天", "整个天空", 0.0, -1.0, "没有硬阴影，物体背光面暗约 1 档"),
    ("室内（窗光 / 吊灯）", "窗或主灯", 0.0, -2.5, "离光源越远越暗：距离翻倍暗 2 档"),
    ("地牢（火把）", "火把 / 法阵", 0.0, -5.0, "火把照到的地方保持正确曝光，其余靠很弱的环境光"),
    ("夜晚户外", "月光 + 灯笼", -1.5, -3.0, "整体比正确曝光暗 1–2 档；灯笼周围按 0 档打"),
]
out["scenes"] = [{"name": n, "key": k, "offset": o, "shadow": s, "note": note,
                  "lit": r1(code(0.18 * 2 ** o, ACES)), "dark": r1(code(0.18 * 2 ** (o + s), ACES)),
                  "lit_f": r1(code(0.18 * 2 ** o, FILMIC)), "dark_f": r1(code(0.18 * 2 ** (o + s), FILMIC)),
                  "lit_le": r1(code(0.18 * 2 ** (o * LE), ACES)), "dark_le": r1(code(0.18 * 2 ** ((o + s) * LE), ACES))}
                 for n, k, o, s, note in SCENES]

# ------------------------------------------------ 8. 灯光换算（曝光 ×1：照到灰卡上 π lux = 正确曝光）
PI = float(np.pi)
cd_2m = PI * 2 ** 2
out["light"] = {
    "pi": round(PI, 2),
    "dir_default": 10, "dir_default_ev": round(float(np.log2(10 / PI)), 1),
    "point_default_lm": round(5000 * 16 * 4 * PI / 1e4, 1), "point_default_cd": round(5000 * 16 / 1e4, 1),
    "point_default_d": round(float(np.sqrt(5000 * 16 / 1e4 / PI)), 1),
    "torch_cd": round(cd_2m, 1), "torch_lm": round(cd_2m * 4 * PI),
    "torch_rows": [{"d": d, "lux": round(cd_2m / d ** 2, 2), "ev": round(float(np.log2(cd_2m / d ** 2 / PI)), 1),
                    "aces": r1(code(0.18 * cd_2m / d ** 2 / PI, ACES))} for d in (1.0, 1.41, 2.0, 2.83, 4.0, 5.66)],
}

# ------------------------------------------------ 9. DCC 对齐：同一个 18% 灰
out["dcc"] = {"ue": r1(code(0.18, ACES)), "ocio": r1(code(0.18, ACES_OCIO)),
              "ocio_plus": r1(code(0.18 * 1.5, ACES_OCIO)), "ev": round(float(np.log2(1.5)), 2)}


# ------------------------------------------------ 10. 小场景渲染：地面 + 灰球 + 黑 / 灰 / 白三张卡 + 阴影里的灰卡（ACES 2.0 + sRGB）
def render(ratio_stops, offset=0.0, W=480, H=320, ss=2):
    """ratio_stops：阴影比阳光下暗几档（光比）。正确曝光定义为：阳光下平放的 18% 灰卡场景值 = 0.18。
    画面右侧是屏幕外一堵墙投下的直边阴影，里面也放了一张灰卡，用来读光比。"""
    w, h = W * ss, H * ss
    cam = np.array([0.0, 4.0, 5.0]); look = np.array([-0.05, 0.25, 0.1])
    fwd = look - cam; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tanf = np.tan(np.radians(30.0) / 2)
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs + 0.5) / w * 2 - 1; v = 1 - (ys + 0.5) / h * 2
    d = fwd[None, None, :] + u[..., None] * tanf * (w / h) * right + v[..., None] * tanf * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = cam
    C = np.array([-0.55, 0.55, -0.6]); R = 0.55                     # 18% 灰球
    oc = o - C
    b = np.einsum("ijk,k->ij", d, oc); cc = oc @ oc - R * R
    disc = b * b - cc
    t_s = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t_s = np.where(t_s > 0, t_s, np.inf)
    t_f = np.where(d[..., 1] < -1e-6, -o[1] / d[..., 1], np.inf)     # 地面 y = 0
    hit_s = t_s < t_f
    t = np.where(hit_s, t_s, t_f)
    P = o + d * t[..., None]
    N = np.where(hit_s[..., None], (P - C) / R, np.array([0.0, 1.0, 0.0]))
    FLOOR = 0.06                                                     # 深色石面
    alb = np.where(hit_s, 0.18, np.full(t.shape, FLOOR))
    for cx, a in ((-1.55, 0.032), (-0.75, 0.18), (0.05, 0.87), (1.15, 0.18)):   # 黑卡 / 灰卡 / 白卡 / 阴影里的灰卡
        m = (~hit_s) & (np.abs(P[..., 0] - cx) < 0.34) & (np.abs(P[..., 2] - 0.9) < 0.34)
        alb = np.where(m, a, alb)
    L = np.array([-0.55, 0.74, 0.38]); L /= np.linalg.norm(L)       # 指向主光
    Ef = PI * 2.0 ** (-ratio_stops)                                  # 平放表面收到的天光照度
    Ek = (PI - Ef) / L[1]                                            # 主光 + 天光照到平放灰卡上 = π lux
    NoL = np.clip(np.einsum("ijk,k->ij", N, L), 0, None)
    oc2 = P + N * 1e-3 - C
    b2 = np.einsum("ijk,k->ij", oc2, L); c2 = np.einsum("ijk,ijk->ij", oc2, oc2) - R * R
    disc2 = b2 * b2 - c2
    sh = ((disc2 > 0) & (-b2 - np.sqrt(np.maximum(disc2, 0)) > 0)) | (c2 < 0)
    wall = np.where(hit_s, 0.0, np.clip((P[..., 0] - 0.55) / 0.08 + 0.5, 0, 1))
    key = np.where(sh, 0.0, Ek * NoL) * (1 - wall)
    sky = Ef * (1 + N[..., 1]) / 2                                   # 均匀天空半球
    bounce = FLOOR * PI * (1 - N[..., 1]) / 2 * 0.5                  # 地面反弹（只影响球的下半部）
    dv = C - P; dist = np.linalg.norm(dv, axis=-1)
    cosv = np.clip(np.einsum("ijk,ijk->ij", dv, N) / np.maximum(dist, 1e-6), 0, 1)
    ao = np.where(hit_s, 1.0, 1 - np.clip((R / np.maximum(dist, R)) ** 2 * cosv, 0, 1))   # 球对地面的解析遮挡
    E = key + (sky + bounce) * ao
    Ls = alb * E / PI * 2.0 ** offset
    img = to8(T.pipeline(np.stack([Ls] * 3, -1), **ACES))
    img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    grey = (~hit_s) & (alb == 0.18)
    return img, float(np.median(Ls[grey & (wall == 0)])), float(np.median(Ls[grey & (wall == 1)]))


out["ratio_imgs"] = []
for r in (1, 2, 3, 5):
    im, lit, dark = render(r)
    Image.fromarray(im).save(os.path.join(IMG, "expo_ratio_%d.png" % r), optimize=True)
    out["ratio_imgs"].append({"stops": r, "ratio": 2 ** r, "lit": r1(code(lit, ACES)), "dark": r1(code(dark, ACES))})
out["bracket_imgs"] = []
for e in (-2, -1, 0, 1, 2):
    im, lit, dark = render(2.5, offset=e)
    name = ("m%d" % -e) if e < 0 else ("p%d" % e)
    Image.fromarray(im).save(os.path.join(IMG, "expo_bracket_%s.png" % name), optimize=True)
    out["bracket_imgs"].append({"ev": e, "file": name, "lit": r1(code(lit, ACES)), "dark": r1(code(dark, ACES)),
                                "white": r1(code(0.87 * 2.0 ** e, ACES))})


# ------------------------------------------------ 11. 特效：火焰渐变（外圈 → 核心）
def lumnorm(c):
    c = np.asarray(c, dtype=np.float64)
    return c / (c @ T.sRGB_2_XYZ[1])


def ramp(colors_at, stops, kw, Wd=720, Hd=40):
    x = np.linspace(0, 1, Wd)
    pos = np.array([p for p, _ in colors_at]); cols = np.array([lumnorm(c) for _, c in colors_at])
    rgb = np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1)
    evs = np.interp(x, [0, 1], stops)
    lin = rgb * (0.18 * np.exp2(evs))[:, None]
    row = to8(T.pipeline(lin, **kw))
    return np.repeat(row[None, :, :], Hd, axis=0)


FX_STOPS = (0.0, 6.0)
NAIVE = [(0.0, (1.0, 0.30, 0.04)), (1.0, (1.0, 0.30, 0.04))]
AUTHORED = [(0.0, (1.0, 0.15, 0.02)), (0.25, (1.0, 0.38, 0.03)), (0.5, (1.0, 0.74, 0.03)),
            (0.72, (1.0, 0.90, 0.30)), (1.0, (1.0, 0.98, 0.90))]
for name, cols, kw in (("naive_filmic", NAIVE, FILMIC), ("naive_aces", NAIVE, ACES), ("authored_aces", AUTHORED, ACES)):
    Image.fromarray(ramp(cols, FX_STOPS, kw)).save(os.path.join(IMG, "fx_ramp_%s.png" % name), optimize=True)
out["fx"] = {"stops": list(FX_STOPS),
             "authored": [{"t": p, "rgb": list(c), "s": [s8(v) for v in c], "hex": hx([s8(v) for v in c])} for p, c in AUTHORED],
             "naive": {"rgb": list(NAIVE[0][1]), "hex": hx([s8(v) for v in NAIVE[0][1]])}}

json.dump(out, open(os.path.join(DATA, "exposure_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("props", "painted", "grey_tol", "marks", "emissive_target", "scenes", "light", "dcc", "ratio_imgs", "bracket_imgs")},
                 ensure_ascii=False, indent=1))
