# -*- coding: utf-8 -*-
"""UE 5.8.2 色调映射（LUT 生成部分）的 numpy 复刻，用于离线对比 Filmic 与 Standard ACES（ACES 2.0）。

对应引擎源码（工程 E:/TorchLight_UE5/Engine/Engine，官方 5.8.2）：
  Shaders/Private/PostProcessCombineLUTs.usf       CombineLUTsCommon / ApplyToneCurve / ComputeFilmColorNoGamma
  Shaders/Private/PostProcessCombineLUTsInner.usf  ComputeOutputACES（ACES 2.0 输出变换，USE_ACES_2）
  Shaders/Private/TonemapCommon.ush                FilmToneMap（Filmic 曲线）
  Shaders/Private/ACES/ACESCommon.ush              矩阵与 RRT 辅助函数
  Shaders/Private/ACES/ACESTonescale.ush           ACES 2.0 tonescale（只用来核对灰阶）
  Source/Runtime/Renderer/Private/PostProcess/PostProcessCombineLUTs.cpp   参数（SDR 编码默认值等）
ACES 2.0 输出变换本身用 OpenColorIO 2.6 的官方内置实现
（ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0），与 UE 移植的是同一份 ACES 2.0 参考算法。

约定：输入是工作色彩空间（sRGB / Rec.709 线性，工程 WorkingColorSpaceChoice=sRGB）的场景线性值，
已经乘过曝光（即进入 Tonemapper 之前的 SceneColor）。输出是 0..1 的「发给显示器的像素值」（编码后）。
默认参数 = UE 默认值（Film Slope 0.88 / Toe 0.55 / Shoulder 0.26 / BlackClip 0 / WhiteClip 0.04，
BlueCorrection 0.6，ExpandGamut 1，ToneCurveAmount 1，Color Grading 全中性，SDR 视口，HDR 输出关闭）。
"""
import numpy as np

# ---------------------------------------------------------------- 矩阵（ACESCommon.ush，按行书写，mul(M, v) = M @ v）
AP0_2_XYZ = np.array([[0.9525523959, 0.0000000000, 0.0000936786],
                      [0.3439664498, 0.7281660966, -0.0721325464],
                      [0.0000000000, 0.0000000000, 1.0088251844]])
XYZ_2_AP0 = np.array([[1.0498110175, 0.0000000000, -0.0000974845],
                      [-0.4959030231, 1.3733130458, 0.0982400361],
                      [0.0000000000, 0.0000000000, 0.9912520182]])
AP1_2_XYZ = np.array([[0.6624541811, 0.1340042065, 0.1561876870],
                      [0.2722287168, 0.6740817658, 0.0536895174],
                      [-0.0055746495, 0.0040607335, 1.0103391003]])
XYZ_2_AP1 = np.array([[1.6410233797, -0.3248032942, -0.2364246952],
                      [-0.6636628587, 1.6153315917, 0.0167563477],
                      [0.0117218943, -0.0082844420, 0.9883948585]])
AP0_2_AP1_MAT = np.array([[1.4514393161, -0.2365107469, -0.2149285693],
                          [-0.0765537734, 1.1762296998, -0.0996759264],
                          [0.0083161484, -0.0060324498, 0.9977163014]])
XYZ_2_sRGB = np.array([[3.2409699419, -1.5373831776, -0.4986107603],
                       [-0.9692436363, 1.8759675015, 0.0415550574],
                       [0.0556300797, -0.2039769589, 1.0569715142]])
sRGB_2_XYZ = np.array([[0.4123907993, 0.3575843394, 0.1804807884],
                       [0.2126390059, 0.7151686788, 0.0721923154],
                       [0.0193308187, 0.1191947798, 0.9505321522]])
D65_2_D60 = np.array([[1.0130349146, 0.0061052578, -0.0149709436],
                      [0.0076982301, 0.9981633521, -0.0050320385],
                      [-0.0028413174, 0.0046851567, 0.9245061375]])
D60_2_D65 = np.array([[0.9872240087, -0.0061132286, 0.0159532883],
                      [-0.0075983718, 1.0018614847, 0.0053300358],
                      [0.0030725771, -0.0050959615, 1.0816806031]])
AP1_RGB2Y = AP1_2_XYZ[1]

# 工作色彩空间（sRGB D65）<-> AP1 / AP0（Bradford 适配，与 FColorSpaceTransform 默认一致）
WCS_2_AP1 = XYZ_2_AP1 @ D65_2_D60 @ sRGB_2_XYZ
AP1_2_WCS = np.linalg.inv(WCS_2_AP1)
WCS_2_AP0 = XYZ_2_AP0 @ D65_2_D60 @ sRGB_2_XYZ
AP1_2_AP0 = XYZ_2_AP0 @ AP1_2_XYZ
AP0_2_AP1 = XYZ_2_AP1 @ AP0_2_XYZ
AP1_2_sRGB = XYZ_2_sRGB @ D60_2_D65 @ AP1_2_XYZ

# ComputeFilmColorNoGamma 里的 Blue Correction
_BlueCorrect = np.array([[0.9404372683, -0.0183068787, 0.0778696104],
                         [0.0083786969, 0.8286599939, 0.1629613092],
                         [0.0005471261, -0.0008833746, 1.0003362486]])
_BlueCorrectInv = np.array([[1.06318, 0.0233956, -0.0865726],
                            [-0.0106337, 1.20632, -0.19569],
                            [-0.000590887, 0.00105248, 0.999538]])
BlueCorrectAP1 = AP0_2_AP1 @ _BlueCorrect @ AP1_2_AP0
BlueCorrectInvAP1 = AP0_2_AP1 @ _BlueCorrectInv @ AP1_2_AP0

# CombineLUTsCommon 里 ExpandGamut 用的「宽色域」
_Wide_2_XYZ = np.array([[0.5441691, 0.2395926, 0.1666943],
                        [0.2394656, 0.7021530, 0.0583814],
                        [-0.0023439, 0.0361834, 1.0552183]])
ExpandMat = (XYZ_2_AP1 @ _Wide_2_XYZ) @ AP1_2_sRGB


def mv(M, x):
    """对 (..., 3) 数组逐像素做 M @ v"""
    return np.einsum("ij,...j->...i", M, x)


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep01(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- 显示编码
def linear_to_srgb(x):
    x = np.maximum(x, 0.0)
    return np.where(x < 0.00313067, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c > 0.04045, np.power((c + 0.055) / 1.055, 2.4), c / 12.92)


def linear_to_bt1886(x):
    """r.LUT.TonemappingSDRInvEOTF=1：OutDeviceColor = pow(OutputGamutColor, 1/2.4)"""
    return np.power(np.maximum(x, 0.0), 1 / 2.4)


def encode(x, eotf):
    """eotf: 'srgb' | 'bt1886'；结果裁到 0..1（8/10 bit 后备缓冲本来就会截断）"""
    e = linear_to_srgb(x) if eotf == "srgb" else linear_to_bt1886(x)
    return np.clip(e, 0.0, 1.0)


# ---------------------------------------------------------------- 共用前段：ExpandGamut + Color Grading
def expand_gamut(ap1, amount=1.0):
    luma = mv(AP1_RGB2Y[None, :], ap1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        chroma = ap1 / luma[..., None]
    d2 = np.sum((chroma - 1) ** 2, axis=-1)
    k = (1 - np.exp2(-4 * d2)) * (1 - np.exp2(-4 * amount * luma * luma))
    k = np.where(np.isfinite(k) & (luma > 0), k, 0.0)
    return lerp(ap1, mv(ExpandMat, ap1), k[..., None])


def _cc(x, sat, con, gam, gain, off):
    luma = mv(AP1_RGB2Y[None, :], x)
    x = np.maximum(0.0, lerp(luma, x, sat))
    x = np.power(x / 0.18, con) * 0.18
    x = np.power(x, 1.0 / gam)
    return x * gain + off


def color_correct_all(ap1, cc=None):
    """PostProcessCombineLUTs.usf ColorCorrectAll。cc 只支持全局项（Global Saturation/Contrast/Gamma/Gain/Offset，
    数值 = RGB 都相同时的 .w 合成值），Shadows/Midtones/Highlights 保持中性；cc=None 时与 UE 默认等价（恒等）。"""
    if not cc:
        return ap1
    sat = cc.get("saturation", 1.0); con = cc.get("contrast", 1.0); gam = cc.get("gamma", 1.0)
    gain = cc.get("gain", 1.0); off = cc.get("offset", 0.0)
    # 三段权重之和恒为 1，三段参数相同 → 等价于直接做一次全局 ColorCorrect
    return _cc(ap1, sat, con, gam, gain, off)


# ---------------------------------------------------------------- Filmic（TonemapCommon.ush FilmToneMap）
def _rgb_2_saturation(rgb):
    mn = np.min(rgb, axis=-1); mx = np.max(rgb, axis=-1)
    return (np.maximum(mx, 1e-10) - np.maximum(mn, 1e-10)) / np.maximum(mx, 1e-2)


def _rgb_2_yc(rgb, w=1.75):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = np.sqrt(np.maximum(b * (b - g) + g * (g - r) + r * (r - b), 0.0))
    return (b + g + r + w * chroma) / 3.0


def _sigmoid_shaper(x):
    t = np.maximum(1 - np.abs(0.5 * x), 0.0)
    y = 1 + np.sign(x) * (1 - t * t)
    return 0.5 * y


def _glow_fwd(yc, gain, mid):
    out = np.where(yc <= 2.0 / 3.0 * mid, gain, np.where(yc >= 2 * mid, 0.0, gain * (mid / np.maximum(yc, 1e-10) - 0.5)))
    return out


def _rgb_2_hue(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2 * r - g - b))
    hue = np.where((r == g) & (g == b), 0.0, hue)
    hue = np.where(hue < 0, hue + 360, hue)
    return np.clip(hue, 0, 360)


def _center_hue(h, c):
    hc = h - c
    hc = np.where(hc < -180, hc + 360, hc)
    hc = np.where(hc > 180, hc - 360, hc)
    return hc


def film_tone_map(ap1, slope=0.88, toe=0.55, shoulder=0.26, black=0.0, white=0.04):
    ap0 = mv(AP1_2_AP0, ap1)
    # Glow
    sat = _rgb_2_saturation(ap0)
    yc = _rgb_2_yc(ap0)
    s = _sigmoid_shaper((sat - 0.4) / 0.2)
    ap0 = ap0 * (1 + _glow_fwd(yc, 0.05 * s, 0.08))[..., None]
    # Red modifier
    hue = _rgb_2_hue(ap0)
    ch = _center_hue(hue, 0.0)
    hw = smoothstep01(1 - np.abs(2 * ch / 135.0)) ** 2
    ap0 = ap0.copy()
    ap0[..., 0] = ap0[..., 0] + hw * sat * (0.03 - ap0[..., 0]) * (1 - 0.82)
    wc = np.maximum(mv(AP0_2_AP1_MAT, ap0), 0.0)
    wc = lerp(mv(AP1_RGB2Y[None, :], wc), wc, 0.96)          # Pre desaturate
    toe_scale = 1 + black - toe
    sh_scale = 1 + white - shoulder
    in_m = out_m = 0.18
    if toe > 0.8:
        toe_match = (1 - toe - out_m) / slope + np.log10(in_m)
    else:
        bt = (out_m + black) / toe_scale - 1
        toe_match = np.log10(in_m) - 0.5 * np.log((1 + bt) / (1 - bt)) * (toe_scale / slope)
    straight_match = (1 - toe) / slope - toe_match
    shoulder_match = shoulder / slope - straight_match
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lc = np.log10(wc)
        straight = slope * (lc + straight_match)
        toe_c = (-black) + (2 * toe_scale) / (1 + np.exp((-2 * slope / toe_scale) * (lc - toe_match)))
        sh_c = (1 + white) - (2 * sh_scale) / (1 + np.exp((2 * slope / sh_scale) * (lc - shoulder_match)))
    toe_c = np.where(lc < toe_match, toe_c, straight)
    sh_c = np.where(lc > shoulder_match, sh_c, straight)
    t = np.clip((lc - toe_match) / (shoulder_match - toe_match), 0, 1)
    if shoulder_match < toe_match:
        t = 1 - t
    t = (3 - 2 * t) * t * t
    tone = lerp(toe_c, sh_c, t)
    tone = np.where(np.isfinite(tone), tone, np.where(lc > 0, 1 + white, -black))
    tone = lerp(mv(AP1_RGB2Y[None, :], tone), tone, 0.93)   # Post desaturate
    return np.maximum(tone, 0.0)


def ue_filmic_linear(x_wcs, film=None, blue=0.6, expand=1.0, tca=1.0, cc=None):
    """返回 Filmic 的显示线性值（sRGB 原色，编码前），对应 ApplyToneCurve 的 FilmColor。"""
    film = film or {}
    ap1 = mv(WCS_2_AP1, np.asarray(x_wcs, dtype=np.float64))
    ap1 = expand_gamut(ap1, expand)
    ap1 = color_correct_all(ap1, cc)
    ap1 = lerp(ap1, mv(BlueCorrectAP1, ap1), blue)
    tm = film_tone_map(ap1, **film)
    ap1 = lerp(ap1, tm, tca)
    ap1 = lerp(ap1, mv(BlueCorrectInvAP1, ap1), blue)
    return np.maximum(mv(AP1_2_WCS, ap1), 0.0)


# ---------------------------------------------------------------- Standard ACES（ACES 2.0，SDR 100 nit Rec.709）
_ocio_proc = None


def _aces2_xyz(aces):
    global _ocio_proc
    if _ocio_proc is None:
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateRaw()
        bt = ocio.BuiltinTransform("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0")
        _ocio_proc = cfg.getProcessor(bt).getDefaultCPUProcessor()
    shp = aces.shape
    buf = np.ascontiguousarray(aces.reshape(-1, 3).astype(np.float32))
    _ocio_proc.applyRGB(buf)
    return buf.reshape(shp).astype(np.float64)


ACES_SCENE_COLOR_MULTIPLIER = 1.5   # r.HDR.Aces.SceneColorMultiplier 默认值；SDR 视口 PaperWhite 203/203 = 1


def ue_aces2_linear(x_wcs, mult=ACES_SCENE_COLOR_MULTIPLIER, expand=1.0, cc=None):
    """返回 Standard ACES（r.HDR.Aces.Version=2）的显示线性值（sRGB 原色，编码前，1.0 = 100 nit）。"""
    ap1 = mv(WCS_2_AP1, np.asarray(x_wcs, dtype=np.float64))
    ap1 = expand_gamut(ap1, expand)
    ap1 = color_correct_all(ap1, cc)
    # ComputeGradedColor → 外层 LUT 把 AP1 线性值裁到 [0, 10000 nit] 后 PQ 编码，内层 LUT 再解回工作色彩空间
    ap1 = np.clip(ap1, 0.0, 100.0)
    graded = mv(AP1_2_WCS, ap1)
    aces = mv(WCS_2_AP0, graded * mult)
    xyz = _aces2_xyz(aces)
    rgb = mv(XYZ_2_sRGB, xyz)
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------- ACES 2.0 tonescale（ACESTonescale.ush 逐行移植，只用于核对）
def aces2_tonescale_nits(x, peak=100.0):
    n = peak; n_r = 100.0; g = 1.15; c = 0.18; c_d = 10.013; w_g = 0.14; t_1 = 0.04
    r_hit = 128. + (896. - 128.) * (np.log(n / n_r) / np.log(10000. / 100.))
    m_0 = n / n_r
    m_1 = 0.5 * (m_0 + np.sqrt(m_0 * (m_0 + 4. * t_1)))
    u = ((r_hit / m_1) / ((r_hit / m_1) + 1)) ** g
    m = m_1 / u
    w_i = np.log(n / 100.) / np.log(2.)
    c_t = c_d / n_r * (1. + w_i * w_g)
    g_ip = 0.5 * (c_t + np.sqrt(c_t * (c_t + 4. * t_1)))
    g_ipp2 = -(m_1 * (g_ip / m) ** (1. / g)) / ((g_ip / m) ** (1. / g) - 1.)
    w_2 = c / g_ipp2
    s_2 = w_2 * m_1
    u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)) ** g
    m_2 = m_1 / u_2
    x = np.asarray(x, dtype=np.float64)
    f = m_2 * (np.maximum(0.0, x) / (x + s_2)) ** g
    h = np.maximum(0., f * f / (f + t_1))
    return h * n_r


# ---------------------------------------------------------------- 管线组合（返回 0..1 像素值）
def pipeline(x_wcs, method, eotf=None, mult=ACES_SCENE_COLOR_MULTIPLIER, cc=None, film=None):
    """method: 'filmic' | 'aces2'；eotf 缺省 = UE 默认（Filmic→sRGB，PPV 选 StandardACES→BT.1886）"""
    if method == "filmic":
        lin = ue_filmic_linear(x_wcs, film=film, cc=cc)
        return encode(lin, eotf or "srgb")
    lin = ue_aces2_linear(x_wcs, mult=mult, cc=cc)
    return encode(lin, eotf or "bt1886")


# ---------------------------------------------------------------- 感知量（OKLab，用于饱和度 / 色相 / 亮度对比）
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab_from_display(code, display="srgb"):
    """code：0..1 像素值。display：观看用的显示器 EOTF（'srgb' 分段 / 'g22' 纯 2.2）。返回 (L, a, b)。"""
    code = np.clip(np.asarray(code, dtype=np.float64), 0, 1)
    lin = srgb_to_linear(code) if display == "srgb" else np.power(code, 2.2)
    lms = mv(_M1, lin)
    lms_ = np.cbrt(lms)
    return mv(_M2, lms_)


def oklch_from_display(code, display="srgb"):
    lab = oklab_from_display(code, display)
    L = lab[..., 0]; C = np.hypot(lab[..., 1], lab[..., 2]); h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    return L, C, h
