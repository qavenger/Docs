# -*- coding: utf-8 -*-
"""核对模型 + 打印关键数字（灰阶）。运行：python -I validate.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ue_tonemap as T

np.set_printoptions(suppress=True, precision=5)

# 1) ACES 2.0：OCIO 参考实现 vs UE ACESTonescale.ush 公式（中性灰，输出亮度 Y）
v = np.logspace(-4, 3, 57)
aces = np.stack([v, v, v], -1)
Y_ocio = T._aces2_xyz(aces)[..., 1]
Y_ts = T.aces2_tonescale_nits(v) / 100.0
rel = np.abs(Y_ocio - Y_ts) / np.maximum(Y_ts, 1e-6)
mask = (Y_ts > 1e-4) & (v < 128)   # 场景值 >= 128 时到峰值：OCIO 裁到 1.0，UE 也在 clamp_zero_to_peakLuminance 里裁
print("[1] ACES2 grey: OCIO vs UE tonescale formula  max rel err (v<128, Y>1e-4) = %.2e" % rel[mask].max())
for x in (0.01, 0.18, 1.0, 10.0, 128.0):
    print("     ACES2065 %.3f -> Y ocio %.5f  formula %.5f" % (x, T._aces2_xyz(np.array([[x, x, x]]))[0, 1], T.aces2_tonescale_nits(x) / 100))

# 2) Filmic：中灰不动点
g = np.array([[0.18, 0.18, 0.18]])
print("[2] Filmic 0.18 grey -> display linear", T.ue_filmic_linear(g)[0], " sRGB code", T.encode(T.ue_filmic_linear(g), "srgb")[0] * 255)

# 3) 灰阶关键数字
ev = np.linspace(-10, 10, 2001)
x = 0.18 * np.exp2(ev)
X = np.stack([x, x, x], -1)
variants = {
    "Filmic (UE 默认)": T.pipeline(X, "filmic")[..., 1],
    "ACES2 默认 (x1.5, BT.1886)": T.pipeline(X, "aces2")[..., 1],
    "ACES2 x1.5 + sRGB 编码": T.pipeline(X, "aces2", eotf="srgb")[..., 1],
    "ACES2 x1.0 + BT.1886": T.pipeline(X, "aces2", mult=1.0)[..., 1],
    "ACES2 x1.0 + sRGB": T.pipeline(X, "aces2", eotf="srgb", mult=1.0)[..., 1],
}
def at(arr, e):
    return np.interp(e, ev, arr)
def first_ev(arr, thr):
    i = np.argmax(arr >= thr)
    return ev[i] if arr[i] >= thr else float("nan")
print("[3] 像素值(0-255) @EV(相对 18%% 灰)   -6     -4     -2     0     +2     +4     +6 | 斜率@0(每档) | 到 90%%/99%%/100%% 白的 EV")
for k, a in variants.items():
    s = (at(a, 0.25) - at(a, -0.25)) / 0.5 * 255
    print("  %-28s" % k, " ".join("%6.1f" % (at(a, e) * 255) for e in (-6, -4, -2, 0, 2, 4, 6)),
          "| %5.1f |" % s, " %.2f / %.2f / %.2f" % (first_ev(a, 0.9), first_ev(a, 0.99), first_ev(a, 0.999)))

# 4) 在 sRGB 显示器上看到的亮度（相对白）：中灰与暗部
print("[4] 在 sRGB 显示器上的显示亮度（%% of white） 中灰 / -3EV / -5EV")
for k, a in variants.items():
    L = T.srgb_to_linear(np.array([at(a, 0), at(a, -3), at(a, -5)]))
    print("  %-28s" % k, " ".join("%7.3f%%" % (l * 100) for l in L))
