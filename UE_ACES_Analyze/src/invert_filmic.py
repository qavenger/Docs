# -*- coding: utf-8 -*-
"""把一张 Filmic 出图（sRGB 8bit 截图）反推成近似的场景线性值，用于「同一画面换成 ACES 会怎样」的近似重映射。

做法：先用灰阶曲线的一维反函数逐通道给初值，再用完整的 Filmic 正向模型（ue_tonemap.ue_filmic_linear，
含 Blue Correction / Glow / Red modifier / 去饱和）做逐通道乘法修正迭代，直到正向结果与截图一致。
限制：截图里已经到 255 的像素（Filmic 顶到白）无法还原真实亮度，只能取曲线到顶的那个值；
UI、调试文字等不经过 Tonemapper 的元素也会被一并当作场景处理，所以截图要裁掉 UI。"""
import numpy as np
import ue_tonemap as T

_v = np.logspace(-7, 3, 4001)
_f = T.ue_filmic_linear(np.stack([_v, _v, _v], -1))[..., 1]   # 灰阶：场景值 → 显示线性
_fmax = _f.max()


def _g(t):
    """灰阶曲线反函数（显示线性 → 场景值），超出曲线上限的取上限"""
    t = np.clip(t, _f[0], _fmax * 0.9995)
    return np.interp(np.log(t), np.log(_f), _v)


def invert_filmic(code_srgb, iters=60):
    """code_srgb：(..., 3) 0..1 的 sRGB 像素值。返回 (场景线性值, 每像素残差(8bit 码值))
    逐通道乘法修正（每步限幅 0.7–1.4 倍防发散），保留每个像素残差最小的那次结果。
    极少数比 Filmic 能输出的颜色还要纯的暗红像素（场景值本身超出 sRGB 色域）无法精确还原，残差会偏大。"""
    code = np.clip(code_srgb, 0, 1)
    target = T.srgb_to_linear(code)
    gt = _g(target)
    x = gt.copy()
    best = x.copy()
    best_res = np.full(code.shape[:-1], np.inf)
    for _ in range(iters):
        fx = T.ue_filmic_linear(x)
        res = np.abs(T.encode(fx, "srgb") - code).max(axis=-1) * 255
        better = res < best_res
        best[better] = x[better]
        best_res[better] = res[better]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(gt > 0, gt / np.maximum(_g(fx), 1e-12), 0.0)
        x = np.clip(x * np.clip(ratio, 0.7, 1.4), 0.0, 1e3)
    return best, best_res
