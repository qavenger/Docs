# -*- coding: utf-8 -*-
"""第二章（PBR 重构）的数值与图解：材质、灯光、物理相机、PPV 过渡、特效贴图：data/exposure_data.json 与 img/expo_*.png、img/fx_*.png。
和 make_assets.py 用同一套 Filmic / ACES 2.0 复刻（ue_tonemap.py），build_html.py 读结果填进页面。

约定：屏幕值都按「曝光后的场景值」计算——正确曝光 = 主光下 18% 灰卡曝光后正好 0.18，此时被主光照亮的漫反射表面 ≈ 它的 BaseColor。
目标流程是手动曝光 + 物理相机：曝光 = 2^(曝光补偿 − EV100)，正确曝光的 EV100 = log2(灰卡照度 ÷ π)（UE 5.8.4 源码核对，见报告附录 A2）。
迁移前的工程现状是曝光锁定 ×1（r.EyeAdaptationQuality=0），相当于 EV100 = 0 的相机。
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

# ------------------------------------------------ 7. 物理曝光：照度 → 正确曝光的 EV100
# UE 的标定（r.EyeAdaptation.LensAttenuation 0.78 + 扩展亮度范围，白点系数 1）：曝光 = 2^(曝光补偿 − EV100)，
# 灰卡亮度 = 0.18 × 照度 ÷ π，曝光后 = 0.18 时 → EV100 = log2(照度 ÷ π)
PI = float(np.pi)


def ev_of_lux(e):
    return float(np.log2(e / PI))


REAL = [("晴天，阳光直射", 32000, 100000), ("全天光、无直射（阴天、树荫）", 10000, 25000), ("很暗的阴天", 100, 100),
        ("办公室照明", 320, 500), ("日出 / 日落时的地面（晴天）", 400, 400), ("居家客厅（澳洲 1998 年调查）", 50, 50),
        ("光线较暗的公共区域", 20, 50), ("满月夜", 0.05, 0.3)]
out["real_lux"] = [{"name": n, "lo": lo, "hi": hi, "ev_lo": round(ev_of_lux(lo), 1), "ev_hi": round(ev_of_lux(hi), 1)} for n, lo, hi in REAL]

# 本项目建议的基准 EV100（相对物理：自然光按真实值，人工光允许提亮，光比保持真实）
SCENES = [
    ("主城白天（晴）", "太阳（天空大气）+ 天光", 14, 15, -2.5, 0.0, "太阳、天空都按真实值；阴影里只有天光"),
    ("阴天 / 雾天", "整个天空", 12, 13, -1.0, 0.0, "没有硬阴影，物体背光面暗约 1 档"),
    ("黄昏", "低角度太阳 + 天光", 10, 12, -3.0, 0.0, "太阳强度不用改，大气会让它变暗变红"),
    ("室内（窗光 / 吊灯）", "窗户或主灯", 7, 8, -2.5, 0.0, "离光源越远越暗：距离翻倍暗 2 档"),
    ("地牢（火盆 / 法阵）", "人工光", 3, 5, -5.0, 0.0, "灯比真实火把亮（相对物理），光比保持真实"),
    ("夜晚户外", "月光 + 灯笼", -2, -1, -3.0, -2.0, "月光提亮到约 1 lux；正常曝光再压 2 档；灯笼按真实流明"),
]
out["scenes"] = []
for n, k, lo, hi, sh, off, note in SCENES:
    mid = (lo + hi) / 2
    out["scenes"].append({"name": n, "key": k, "ev_lo": lo, "ev_hi": hi, "shadow": sh, "offset": off, "note": note,
                          "lux_lo": PI * 2 ** lo, "lux_hi": PI * 2 ** hi, "shutter": 2 ** mid / 16,
                          "lit": r1(code(0.18 * 2 ** off, ACES)), "dark": r1(code(0.18 * 2 ** (off + sh), ACES))})

# 物理相机：ISO 100、光圈 f/4 固定时，EV100 对应的快门（Shutter Speed 一栏填的是 1/秒 的分母）
out["shutter_f4"] = [{"ev": e, "s": 2.0 ** e / 16} for e in range(-2, 16)]

# 实验：晴天 16 法则（f/16、1/125 秒、ISO 100）+ 10 万 lux 的太阳
ev_s16 = float(np.log2(16 ** 2 * 125))
post = 0.18 * 100000 / PI / 2 ** ev_s16
out["sunny16"] = {"ev": round(ev_s16, 2), "nits": round(0.18 * 100000 / PI), "pre": 2 ** -ev_s16, "post": round(post, 3),
                  "aces": r1(code(post, ACES)), "final": final_lin(post, ACES),
                  "half": r1(code(post / 2, ACES)), "double": r1(code(post * 2, ACES))}

# 真实灯具（流明）：1 / 2 / 3 米处的照度与“正确曝光”EV100
LAMPS = [("蜡烛", 4 * PI), ("40 W 白炽灯", 450), ("60 W 白炽灯", 800), ("100 W 白炽灯", 1600)]
out["lamps"] = [{"name": n, "lm": lm, "cd": lm / (4 * PI),
                 "rows": [{"d": d, "lux": lm / (4 * PI) / d ** 2, "ev": round(ev_of_lux(lm / (4 * PI) / d ** 2), 1)} for d in (1, 2, 3)]}
                for n, lm in LAMPS]
# 例：地牢基准 EV100 4，玩家活动区离火盆 3 米 → 火盆要多少流明
brazier_cd = PI * 2 ** 4 * 3 ** 2
out["brazier"] = {"ev": 4, "d": 3, "lux": PI * 2 ** 4, "cd": brazier_cd, "lm": brazier_cd * 4 * PI}
# 离火盆越远越暗：距离每 ×√2 暗 1 档（灰卡相对地牢基准的档数与屏幕值）
out["brazier"]["rows"] = [{"d": d, "lux": brazier_cd / d ** 2, "ev": round(float(np.log2(brazier_cd / d ** 2 / (PI * 2 ** 4))), 2),
                           "aces": r1(code(0.18 * brazier_cd / d ** 2 / (PI * 2 ** 4), ACES))} for d in (1.5, 2.12, 3, 4.24, 6)]

# 亮度参考（cd/m²，Wikipedia「Orders of magnitude (luminance)」）
out["nits_ref"] = [("正午的太阳圆盘", 1.6e9), ("磨砂白炽灯泡", 1.3e5), ("荧光灯管", 1.2e4), ("正午晴空", 1.0e4), ("月球表面", 2.5e3),
                   ("正午阴天的天空", 1.0e3), ("普通 LCD 显示器峰值", 250), ("sRGB 标准观看环境里的显示器白", 80)]
# 自发光：想比灰卡亮 N 档，在不同场景 EV100 下要填多少尼特
out["emit_nits"] = [{"ev": e, "rows": [round(0.18 * 2 ** (e + n)) if 0.18 * 2 ** (e + n) >= 10 else round(0.18 * 2 ** (e + n), 2) for n in (0, 3, 6)]}
                    for e in (15, 7, 4, -1.7)]

# ------------------------------------------------ 8. 旧工程（曝光锁定 ×1）里的灯光量级，用于说明迁移差距
cd_2m = PI * 2 ** 2
out["light"] = {
    "pi": round(PI, 2),
    "dir_default": 10, "dir_default_ev": round(float(np.log2(10 / PI)), 1),
    "point_default_lm": round(5000 * 16 * 4 * PI / 1e4, 1), "point_default_cd": round(5000 * 16 / 1e4, 1),
    "point_default_d": round(float(np.sqrt(5000 * 16 / 1e4 / PI)), 1),
    "gap_ev": round(ev_of_lux(100000), 1),
}

# ------------------------------------------------ 9. DCC 对齐：同一个 18% 灰
out["dcc"] = {"ue": r1(code(0.18, ACES)), "ocio": r1(code(0.18, ACES_OCIO)),
              "ocio_plus": r1(code(0.18 * 1.5, ACES_OCIO)), "ev": round(float(np.log2(1.5)), 2)}

# ------------------------------------------------ 9b. 场景过渡：户外（EV100 14）走进山洞（EV100 7），PPV 混合哪个参数
# 曝光补偿按档线性插值；快门、ISO 按数值线性插值（PostProcessUtils / SceneView 的 LERP_PP），换算成档就很不均匀
tt = np.linspace(0, 1, 101)
out["trans"] = {"t": [round(float(v) * 100, 1) for v in tt],
                "ec": [round(float(v), 3) for v in 7 * tt],
                "shutter": [round(float(v), 3) for v in 14 - np.log2(16 * ((1 - tt) * 1024 + tt * 8))],
                "iso": [round(float(v), 3) for v in 14 - np.log2(16 * 1024 * 100 / ((1 - tt) * 100 + tt * 12800))]}
i50 = 50
out["trans_mid"] = {"ec": out["trans"]["ec"][i50], "shutter": out["trans"]["shutter"][i50], "iso": out["trans"]["iso"][i50],
                    "shutter_90": out["trans"]["shutter"][90]}


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


# ------------------------------------------------ 11. 特效：Filmic 时代的火焰贴图 → 反推 ACES 用的新贴图
# 做法：旧贴图 = 灰度火焰 × 单一橙色，材质 Emissive = 贴图 × 强度。先算出它在 Filmic 下的显示颜色（目标），
# 再沿亮度逐级找出 ACES 下能显示成同样颜色的场景值（OKLab 最小二乘，批量 LM），得到一条新的颜色渐变；
# 新贴图 = 同一张灰度图套这条渐变，强度系数按新渐变的最大值重新定。
FW, FH = 300, 400


def value_noise(shape, cells, seed):
    rng = np.random.default_rng(seed)
    gy, gx = cells
    g = rng.random((gy + 3, gx + 3))
    h, w = shape
    ys = np.linspace(0, gy, h, endpoint=False)[:, None] + 1
    xs = np.linspace(0, gx, w, endpoint=False)[None, :] + 1
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    fy = ys - y0; fx = xs - x0
    sy = fy * fy * (3 - 2 * fy); sx = fx * fx * (3 - 2 * fx)
    a = g[y0, x0]; b = g[y0, x0 + 1]; c = g[y0 + 1, x0]; d = g[y0 + 1, x0 + 1]
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def fbm(shape, base, seed, octaves=4):
    v = np.zeros(shape); amp = 1.0; tot = 0.0
    for o in range(octaves):
        v += amp * value_noise(shape, (base[0] * 2 ** o, base[1] * 2 ** o), seed + o)
        tot += amp; amp *= 0.5
    return v / tot


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def flame_mask():
    """程序生成的火焰灰度图（线性亮度 0–1），相当于美术画的火焰贴图的亮度"""
    yy, xx = np.mgrid[0:FH, 0:FW]
    x = (xx + 0.5) / FW * 2 - 1
    y = (FH - yy - 0.5) / FH * 1.30 - 0.08
    yc = np.clip(y, 0, None)
    nw = fbm((FH, FW), (3, 2), 5) - 0.5
    n2 = fbm((FH, FW), (7, 4), 23) - 0.5
    n3 = fbm((FH, FW), (16, 9), 37) - 0.5
    xw = x + 0.30 * nw * yc ** 1.2 + 0.07 * np.sin(3.4 * y + 0.7) * yc
    wid = 0.44 * (1 - 0.70 * np.clip(y / 1.1, 0, 1)) + 0.03
    shape = np.exp(-(xw / wid) ** 2)
    d = shape * smoothstep(-0.08, 0.08, y) * (1.15 - 0.55 * yc) + 0.55 * n2 * (0.3 + yc) + 0.22 * n3 * (0.2 + yc) - 0.12
    f = smoothstep(0.06, 0.95, d)
    hot = np.exp(-(xw / 0.22) ** 2 - ((y - 0.16) / 0.32) ** 2)
    return np.clip(f ** 2.3 * (0.45 + 0.75 * hot), 0, 1)


def solve_aces(tgt_lab, x0, iters=60):
    """找场景线性值 x ≥ 0，让 ACES(x) 的 OKLab 尽量接近目标（逐样本批量 Levenberg–Marquardt，log 空间）"""
    u = np.log(np.maximum(x0, 1e-6))
    lam = np.full(len(u), 1e-2)

    def f(uu):
        return T.oklab_from_display(T.pipeline(np.exp(uu), **ACES))

    cur = f(u); err = np.sum((cur - tgt_lab) ** 2, -1)
    for _ in range(iters):
        J = np.zeros((len(u), 3, 3)); hstep = 1e-3
        for k in range(3):
            du = np.zeros(3); du[k] = hstep
            J[:, :, k] = (f(u + du) - cur) / hstep
        r = cur - tgt_lab
        A = np.einsum("nik,nil->nkl", J, J) + lam[:, None, None] * np.eye(3)[None]
        step = np.clip(-np.linalg.solve(A, np.einsum("nik,ni->nk", J, r)[..., None])[..., 0], -1.5, 1.5)
        un = u + step; new = f(un); en = np.sum((new - tgt_lab) ** 2, -1)
        ok = en < err
        u = np.where(ok[:, None], un, u); cur = np.where(ok[:, None], new, cur); err = np.where(ok, en, err)
        lam = np.where(ok, lam * 0.4, lam * 4)
    return np.exp(u)


with np.errstate(invalid="ignore", divide="ignore"):
    FX_TINT = T.srgb_to_linear(np.array([255, 128, 24]) / 255.0)    # 旧贴图的单一橙色
    FX_I_OLD = 26.0                                                 # 旧材质：Emissive = 贴图 × 26
    mask = flame_mask()
    lv = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), 0, 1023))])   # 沿 log 亮度取样，暗部不出色带
    old_lut = lv[:, None] * FX_TINT
    tgt = T.pipeline(old_lut * FX_I_OLD, **FILMIC)
    tgt_lab = T.oklab_from_display(tgt)
    x_lut = solve_aces(tgt_lab, old_lut * FX_I_OLD)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    for _ in range(2):                                              # 中值去掉个别跳到别的解的样本，再精修
        lu = np.median(swv(np.pad(np.log(np.maximum(x_lut, 1e-7)), ((3, 3), (0, 0)), mode="edge"), 7, axis=0), axis=-1)
        x_lut = solve_aces(tgt_lab, np.exp(lu), iters=40)
    ll = np.log(np.maximum(lv, 1e-4))
    wts = np.exp(-0.5 * ((ll[:, None] - ll[None, :]) / 0.35) ** 2); wts /= wts.sum(1, keepdims=True)
    x_lut = np.exp(wts @ np.log(np.maximum(x_lut, 1e-7)))            # 沿亮度平滑，像手绘渐变一样顺
    x_lut[0] = 0
    lut_de = np.sqrt(np.sum((T.oklab_from_display(T.pipeline(x_lut, **ACES)) - tgt_lab) ** 2, -1)) * 100
    FX_I_NEW = float(x_lut.max())                                   # 新贴图最亮处 = 1，强度系数随之降低
    new_lut = x_lut / FX_I_NEW

    def apply_lut(lut, b):
        lb = np.log(np.maximum(b, 1e-4))
        return np.stack([np.interp(lb, ll, lut[:, k]) for k in range(3)], -1) * (b > 0)[..., None]

    old_tex = mask[..., None] * FX_TINT
    new_tex = apply_lut(new_lut, mask)
    shots = {"old_tex": to8(T.linear_to_srgb(old_tex)), "old_filmic": to8(T.pipeline(old_tex * FX_I_OLD, **FILMIC)),
             "old_aces": to8(T.pipeline(old_tex * FX_I_OLD, **ACES)), "new_tex": to8(T.linear_to_srgb(new_tex)),
             "new_aces": to8(T.pipeline(new_tex * FX_I_NEW, **ACES))}
    split = shots["old_filmic"].copy(); split[:, FW // 2:] = shots["new_aces"][:, FW // 2:]; split[:, FW // 2 - 1:FW // 2 + 1] = 255
    shots["split"] = split
    for k, im in shots.items():
        Image.fromarray(im).save(os.path.join(IMG, "fx_%s.png" % k), optimize=True)
    a = shots["old_filmic"] / 255.0; c = shots["new_aces"] / 255.0; o = shots["old_aces"] / 255.0
    m = mask > 0.01
    img_de = np.linalg.norm(T.oklab_from_display(a) - T.oklab_from_display(c), axis=-1) * 100
    img_de_old = np.linalg.norm(T.oklab_from_display(a) - T.oklab_from_display(o), axis=-1) * 100
    # 渐变贴图（Gradient Map）：横轴 = 灰度贴图的像素值（PS 0–255）
    rs = T.srgb_to_linear(np.arange(256) / 255.0)
    Image.fromarray(np.repeat(to8(T.linear_to_srgb(rs[:, None] * FX_TINT))[None], 28, 0)).save(os.path.join(IMG, "fx_ramp_old.png"), optimize=True)
    Image.fromarray(np.repeat(to8(T.linear_to_srgb(apply_lut(new_lut, rs))) [None], 28, 0)).save(os.path.join(IMG, "fx_ramp_new.png"), optimize=True)

STOPS = (40, 64, 96, 128, 160, 192, 224, 255)
out["fx"] = {"i_old": FX_I_OLD, "i_new": round(FX_I_NEW, 1), "tint": [255, 128, 24], "tint_hex": hx([255, 128, 24]),
             "lut_de": round(float(lut_de[1:].mean()), 2), "img_de": round(float(img_de[m].mean()), 2),
             "img_de95": round(float(np.percentile(img_de[m], 95)), 2), "img_de_old": round(float(img_de_old[m].mean()), 1),
             "stops": [{"g": g, "old": hx(to8(T.linear_to_srgb(T.srgb_to_linear(np.array(g / 255.0)) * FX_TINT))),
                        "new": hx(to8(T.linear_to_srgb(apply_lut(new_lut, np.array([T.srgb_to_linear(np.array(g / 255.0))]))[0])))} for g in STOPS]}

json.dump(out, open(os.path.join(DATA, "exposure_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("props", "marks", "scenes", "sunny16", "brazier", "trans_mid", "dcc", "fx")}, ensure_ascii=False, indent=1))
