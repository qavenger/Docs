# -*- coding: utf-8 -*-
"""把 data/*.json 注入 report_template.html，生成 ../index.html。
模板里的 {{name}} 由下面的 V 字典替换（数字全部从数据算出来，避免手抄出错；h_ 开头的是由数据生成的 HTML 片段）；
/*__REPORT_DATA__*/ 替换成图表用的 JSON。先跑 make_assets.py、make_exposure.py 和 make_cvars.py。运行：python -I build_html.py"""
import html, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
D = json.load(open(os.path.join(ROOT, "data", "report_data.json"), encoding="utf-8"))
FIT = json.load(open(os.path.join(ROOT, "data", "fit.json"), encoding="utf-8"))
SCAN = json.load(open(os.path.join(ROOT, "data", "project_scan.json"), encoding="utf-8"))

ts = D["tone_stats"]
f, ad, as_, ac = ts["filmic"], ts["aces_default"], ts["aces_srgb"], ts["aces_comp"]
ck = D["checker"]
sc = D["scenes"]
hue = D["hue"]
comp = FIT["fit_srgb_rounded"]
b2 = FIT["fit_bt1886_rounded"]


def n(v, nd=1):
    s = ("%." + str(nd) + "f") % v
    return s.replace("-", "−")


def sgn(v, nd=1):
    s = n(abs(v), nd)
    return ("+" if v > 0 else "−" if v < 0 else "") + s


V = {
    "mid_f": n(f["mid"]), "mid_ad": n(ad["mid"]), "mid_as": n(as_["mid"]), "mid_ac": n(ac["mid"]),
    "mid_f0": n(f["mid"], 0), "mid_ad0": n(ad["mid"], 0),
    "slope_f": n(f["slope"]), "slope_ad": n(ad["slope"]), "slope_as": n(as_["slope"]), "slope_ac": n(ac["slope"]),
    "slope_drop_ad": n((1 - ad["slope"] / f["slope"]) * 100, 0), "slope_drop_as": n((1 - as_["slope"] / f["slope"]) * 100, 0),
    "m3_f": n(f["lum_m3"], 2), "m3_ad": n(ad["lum_m3"], 2), "m3_as": n(as_["lum_m3"], 2),
    "m5_f": n(f["lum_m5"], 2), "m5_ad": n(ad["lum_m5"], 2), "m5_as": n(as_["lum_m5"], 2),
    "m3_ratio": n(ad["lum_m3"] / f["lum_m3"], 1), "m5_ratio": n(ad["lum_m5"] / f["lum_m5"], 1),
    "cm2_f": n(f["code_m2"]), "cm2_ad": n(ad["code_m2"]), "cm2_as": n(as_["code_m2"]), "cm2_ac": n(ac["code_m2"]),
    "cm4_f": n(f["code_m4"]), "cm4_ad": n(ad["code_m4"]), "cm4_as": n(as_["code_m4"]),
    "cp2_f": n(f["code_p2"]), "cp2_ad": n(ad["code_p2"]), "cp2_as": n(as_["code_p2"]), "cp2_ac": n(ac["code_p2"]),
    "ev90_f": sgn(f["ev90"]), "ev90_a": sgn(as_["ev90"]), "ev99_f": sgn(f["ev99"]), "ev99_a": sgn(as_["ev99"]),
    "sc90_f": n(0.18 * 2 ** f["ev90"], 1), "sc90_a": n(0.18 * 2 ** as_["ev90"], 1),
    "sc99_f": n(0.18 * 2 ** f["ev99"], 1), "sc99_a": n(0.18 * 2 ** as_["ev99"], 0),
    "x99_f": n(2 ** f["ev99"], 0), "x99_a": n(2 ** as_["ev99"], 0),
    "chk_f": n(ck["filmic"]["C_mean"]), "chk_ad": n(ck["aces_default"]["C_mean"]), "chk_as": n(ck["aces_srgb"]["C_mean"]),
    "chk_ac": n(ck["aces_comp"]["C_mean"]),
    "chk_drop_ad": n((1 - ck["aces_default"]["C_mean"] / ck["filmic"]["C_mean"]) * 100, 0),
    "chk_drop_as": n((1 - ck["aces_srgb"]["C_mean"] / ck["filmic"]["C_mean"]) * 100, 0),
    "de_ad": n(FIT["aces_default"]), "de_as": n(FIT["aces_srgb"]), "de_ac": n(comp["dE"]), "de_b2": n(b2["dE"]),
    "comp_ev": sgn(comp["ev"]), "comp_con": "%.2f" % comp["contrast"], "comp_sat": "%.2f" % comp["saturation"],
    "comp_mult": n(1.5 * 2 ** comp["ev"], 1),
    "b2_ev": sgn(b2["ev"]), "b2_con": ("%.2f" % b2["contrast"]).rstrip("0").rstrip("."), "b2_sat": "%.2f" % b2["saturation"],
    "b2_mult": n(1.5 * 2 ** b2["ev"], 1),
    "dg_ad": n(sc["dungeon"]["aces_default"]["dE"]), "dg_as": n(sc["dungeon"]["aces_srgb"]["dE"]), "dg_ac": n(sc["dungeon"]["aces_comp"]["dE"]),
    "sr_ad": n(sc["selectrole"]["aces_default"]["dE"]), "sr_as": n(sc["selectrole"]["aces_srgb"]["dE"]), "sr_ac": n(sc["selectrole"]["aces_comp"]["dE"]),
    "ckde_ad": n(D["checker_dE"]["aces_default"]), "ckde_as": n(D["checker_dE"]["aces_srgb"]), "ckde_ac": n(D["checker_dE"]["aces_comp"]),
    "dg_share": n((1 - sc["dungeon"]["aces_srgb"]["dE"] / sc["dungeon"]["aces_default"]["dE"]) * 100, 0),
    "sr_share": n((1 - sc["selectrole"]["aces_srgb"]["dE"] / sc["selectrole"]["aces_default"]["dE"]) * 100, 0),
    "res_dg": n(sc["dungeon"]["res_gt2"], 2), "res_sr": n(sc["selectrole"]["res_gt2"], 0),
    "hue_or_f": n(hue["filmic"][0], 0), "hue_bl_f": n(hue["filmic"][1], 0), "hue_rd_f": n(hue["filmic"][2], 0),
    "hue_pu_f": n(hue["filmic"][3], 0), "hue_max_a": n(max(hue["aces_srgb"]), 0),
    "scan_film_ship": str(SCAN["film_any"]["shipping"]), "scan_film_all": str(SCAN["film_any"]["all"]),
    "scan_sat_ship": str(SCAN["bOverride_ColorSaturation"]["shipping"]),
    "scan_exp_ship": str(SCAN["bOverride_AutoExposureBias"]["shipping"]),
    "scan_con_ship": str(SCAN["bOverride_ColorContrast"]["shipping"]),
}
for i, row in enumerate(D["ev_match"]):
    for k in ("filmic", "aces_default", "aces_srgb", "aces_comp"):
        V["evm_%d_%s" % (row["code"], k)] = sgn(row[k]) if row[k] is not None else "—"
    if row["filmic"] is not None and row["aces_srgb"] is not None:
        V["evm_%d_d" % row["code"]] = sgn(row["aces_srgb"] - row["filmic"])
        V["evm_%d_x" % row["code"]] = n(2 ** (row["aces_srgb"] - row["filmic"]), 1)
for name in ("dungeon", "selectrole"):
    for k in ("filmic", "aces_default", "aces_srgb", "aces_comp"):
        s = sc[name][k]
        V["%s_%s_p05" % (name, k)] = n(s["L_p05"] * 100)
        V["%s_%s_p95" % (name, k)] = n(s["L_p95"] * 100)
        V["%s_%s_C" % (name, k)] = n(s["C_mean"], 2)
        V["%s_%s_dE" % (name, k)] = n(s["dE"])

# ---------------------------------------------------------------- 第二章：材质、灯光、曝光、特效的数值与图解（make_exposure.py）
EX = json.load(open(os.path.join(ROOT, "data", "exposure_data.json"), encoding="utf-8"))
SCAN2 = json.load(open(os.path.join(ROOT, "data", "project_scan_exposure.json"), encoding="utf-8"))
pr, mk, lt = EX["props"], EX["marks"], EX["light"]


def i0(v):
    return "%d" % int(round(v))


def esc(s):
    return html.escape(str(s), quote=True)


def grey_hex(code):
    c = max(0, min(255, int(round(code))))
    return "#%02X%02X%02X" % (c, c, c)


def chip(hexcol, cls="c8"):
    return '<span class="%s" style="background:%s"></span>' % (cls, hexcol)


def lin3(v):
    return ", ".join(("%.3f" % x).rstrip("0").rstrip(".") if x not in (0, 1) else "%d" % x for x in v)


tol = EX["grey_tol"]
emi = {e["stop"]: e for e in EX["emissive"]}
et = {e["code"]: e for e in EX["emissive_target"]}
le = EX["local_exposure"]
V.update({
    "x_grey_aces": i0(pr["grey"]["aces"]), "x_grey_filmic": i0(pr["grey"]["filmic"]),
    "x_white_aces": i0(pr["white"]["aces"]), "x_white_filmic": i0(pr["white"]["filmic"]),
    "x_black_aces": i0(pr["black"]["aces"]), "x_black_filmic": i0(pr["black"]["filmic"]),
    "x_one_aces": i0(pr["max"]["aces"]), "x_one_filmic": i0(pr["max"]["filmic"]),
    "x_black_lin": "%.3f" % pr["black"]["lin"], "x_floor_lin": "%.3f" % pr["floor_tol"]["lin"],
    "x_black_ev": sgn(mk["black"]), "x_white_ev": sgn(mk["white"]), "x_one_ev": sgn(mk["one"]),
    "x_aces90": sgn(mk["aces90"]), "x_aces99": sgn(mk["aces99"]), "x_filmic90": sgn(mk["filmic90"]), "x_filmic99": sgn(mk["filmic99"]),
    "x_grey_final_aces": "%.3f" % EX["grey_final_aces"], "x_grey_final_filmic": "%.3f" % EX["grey_final_filmic"],
    "x_white_final_aces": "%.2f" % EX["white_final"]["aces"], "x_white_final_filmic": "%.2f" % EX["white_final"]["filmic"],
    "x_double_aces": i0(emi[1]["aces"]), "x_double_filmic": i0(emi[1]["filmic"]),
    "x_tol_lo": i0(tol[0]["aces"]), "x_tol_hi": i0(tol[2]["aces"]), "x_tol_lo_ev": sgn(tol[0]["ev"], 2), "x_tol_hi_ev": sgn(tol[2]["ev"], 2),
    "x_ps118_naive": "%.3f" % EX["ps118_naive"], "x_ps118_naive_ev": sgn(EX["ps118_naive_ev"]),
    "x_ps128_lin": "%.3f" % EX["ps128_lin"], "x_lin05_s": str(EX["lin05_s"]),
    "x_dir_default_ev": sgn(lt["dir_default_ev"]), "x_point_lm": i0(lt["point_default_lm"]), "x_point_cd": i0(lt["point_default_cd"]),
    "x_point_d": n(lt["point_default_d"]), "x_gap_ev": n(lt["gap_ev"], 0),
    "x_dcc_ue": i0(EX["dcc"]["ue"]), "x_dcc_ocio": i0(EX["dcc"]["ocio"]), "x_dcc_ev": "%.2f" % EX["dcc"]["ev"],
    "x_le_sh_aces": i0(le[1]["aces"]), "x_le_sh_aces_le": i0(le[1]["aces_le"]), "x_le_dg_aces": i0(le[0]["aces"]),
    "x_le_dg_aces_le": i0(le[0]["aces_le"]), "x_le_w_aces": i0(le[2]["aces"]), "x_le_w_aces_le": i0(le[2]["aces_le"]),
    "x_et230_a": sgn(et[230]["aces"]), "x_et230_f": sgn(et[230]["filmic"]), "x_et230_av": "%.2f" % et[230]["aces_v"], "x_et230_fv": "%.2f" % et[230]["filmic_v"],
    "s2_bias_all": str(SCAN2["bOverride_AutoExposureBias"]["all"]), "s2_bias_ship": str(SCAN2["bOverride_AutoExposureBias"]["shipping"]),
    "s2_method_all": str(SCAN2["bOverride_AutoExposureMethod"]["all"]), "s2_method_ship": str(SCAN2["bOverride_AutoExposureMethod"]["shipping"]),
    "s2_phys_all": str(SCAN2["bOverride_AutoExposureApplyPhysicalCameraExposure"]["all"]),
    "s2_phys_ship": str(SCAN2["bOverride_AutoExposureApplyPhysicalCameraExposure"]["shipping"]),
    "s2_le_all": str(SCAN2["bOverride_LocalExposureHighlightContrastScale"]["all"] + SCAN2["bOverride_LocalExposureShadowContrastScale"]["all"]),
    "s2_date": SCAN2["date"],
})


def sw_grey(item, extra="", cls="sw"):
    return ('<div class="%s">%s<b>%s</b><span class="v">UE <b>%s</b></span><span class="v">PS %d · %s</span>%s</div>'
            % (cls, chip(item["hex"], "chip"), esc(item["name"]), ("%.3f" % item["lin"]).rstrip("0").rstrip("."), item["s"], item["hex"], extra))


nm = sorted(EX["nonmetal"] + [dict(pr["grey"], name="18% 灰卡")], key=lambda x: x["lin"])
V["h_nonmetal"] = '<div class="sws">' + "".join(
    sw_grey(x, '<span class="v tag">基准</span>' if x["name"] == "18% 灰卡" else "", "sw ref" if x["name"] == "18% 灰卡" else "sw") for x in nm) + "</div>"
V["h_metal"] = '<div class="sws">' + "".join(
    '<div class="sw">%s<b>%s</b><span class="v">UE <b>%s</b></span><span class="v">PS %s · %s</span></div>'
    % (chip(m["hex"], "chip"), esc(m["name"]), lin3(m["lin"]), ",".join(str(c) for c in m["s"]), m["hex"]) for m in EX["metal"]) + "</div>"
V["h_checker"] = '<div class="ccgrid">' + "".join(
    '<div class="cc">%s<span>%s</span></div>' % (chip(c["hex"], "chip"), esc(c["name"])) for c in EX["checker_ref"]) + "</div>"
V["h_checker_rows"] = "".join(
    '<tr><td>%s%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td></tr>'
    % (chip(c["hex"]), esc(c["name"]), ", ".join(str(v) for v in c["s"]), c["hex"], lin3(c["lin"])) for c in EX["checker_ref"])
V["h_painted_rows"] = "".join(
    '<tr><td>%s<b>%d</b>（UE %s）</td><td class="num">%s%s</td><td class="num">%s%s</td></tr>'
    % (chip(grey_hex(p["s"])), p["s"], ("%.3f" % p["lin"]).rstrip("0").rstrip("."), chip(grey_hex(p["filmic"])), i0(p["filmic"]),
       chip(grey_hex(p["aces"])), i0(p["aces"])) for p in EX["painted"])
V["h_final_rows"] = "".join('<tr><td class="num">%s</td><td class="num">%s%d</td></tr>' % (("%.3f" % f["lin"]).rstrip("0").rstrip("."), chip(grey_hex(f["s"])), f["s"])
                            for f in EX["final_color"])
V["h_scene_rows"] = "".join(
    '<tr><td>%s</td><td>%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s%s / %s%s</td><td>%s</td></tr>'
    % (esc(s["name"]), esc(s["key"]), ("0（= 正确曝光）" if s["offset"] == 0 else sgn(s["offset"]) + " 档"), sgn(s["shadow"]) + " 档",
       chip(grey_hex(s["lit"])), i0(s["lit"]), chip(grey_hex(s["dark"])), i0(s["dark"]), esc(s["note"])) for s in EX["scenes"])
V["h_emissive_rows"] = "".join(
    '<tr><td class="num">%s 档</td><td class="num"><b>%s</b></td><td class="num">%s%s</td><td class="num">%s%s</td></tr>'
    % ("+%d" % e["stop"] if e["stop"] else "0", ("%.2f" % e["value"]).rstrip("0").rstrip("."), chip(grey_hex(e["filmic"])), i0(e["filmic"]), chip(grey_hex(e["aces"])), i0(e["aces"]))
    for e in EX["emissive"])
V["h_emit_target_rows"] = "".join(
    '<tr><td class="num">%s%d</td><td class="num">%s 档（Emissive %s）</td><td class="num">%s 档（Emissive %s）</td></tr>'
    % (chip(grey_hex(t["code"])), t["code"], sgn(t["filmic"]), "%.2f" % t["filmic_v"], sgn(t["aces"]), "%.2f" % t["aces_v"]) for t in EX["emissive_target"])
RATIO_NOTE = {1: "阴天、柔光、补光很足", 2: "晴天里阴影偏亮（有大面积反光）", 3: "晴天直射、室内窗光", 5: "地牢、夜晚火把"}
V["h_ratio"] = '<div class="tiles">' + "".join(
    '<figure class="tile"><img src="img/expo_ratio_%d.png" alt="光比 %d 档：阴影里的灰卡比阳光下暗 %d 档" loading="lazy" width="480" height="320">'
    '<figcaption><b>光比 %d 档（%d : 1）</b>亮处灰卡 %s → 阴影灰卡 %s%s<br><span class="muted">%s</span></figcaption></figure>'
    % (r["stops"], r["stops"], r["stops"], r["stops"], r["ratio"], i0(r["lit"]), chip(grey_hex(r["dark"])), i0(r["dark"]), RATIO_NOTE[r["stops"]])
    for r in EX["ratio_imgs"]) + "</div>"
V["h_bracket"] = '<div class="tiles five">' + "".join(
    '<figure class="tile%s"><img src="img/expo_bracket_%s.png" alt="曝光 %s 档" loading="lazy" width="480" height="320">'
    '<figcaption><b>%s</b>亮处灰卡 %s · 白卡 %s</figcaption></figure>'
    % (" ref" if b["ev"] == 0 else "", b["file"], sgn(b["ev"], 0) if b["ev"] else "0",
       ("0 档 · 正确曝光" if b["ev"] == 0 else sgn(b["ev"], 0) + " 档"), i0(b["lit"]), i0(b["white"])) for b in EX["bracket_imgs"]) + "</div>"
V["h_checker_mini"] = "".join('<i style="background:%s"></i>' % c["hex"] for c in EX["checker_ref"])
V["x_grey_hex"] = pr["grey"]["hex"]
V["x_grey_screen_pct"] = n(EX["grey_final_aces"] * 100)
V["x_grey_aces_chip"] = chip(grey_hex(EX["dcc"]["ue"]))
V["x_dcc_ocio_chip"] = chip(grey_hex(EX["dcc"]["ocio"]))
V["x_et230_d"] = sgn(et[230]["aces"] - et[230]["filmic"])
V["x_et245_d"] = sgn(et[245]["aces"] - et[245]["filmic"])
V["comp_gain"] = "%.2f" % (2 ** comp["ev"])

# ---------------------------------------------------------------- 第二章：物理相机、灯光、过渡、特效
PI_ = 3.141592653589793
SKY = SCAN2["components"]
s16, br, tm = EX["sunny16"], EX["brazier"], EX["trans_mid"]


def evs(v):
    """EV100 / 档数：整数不带小数，负号用 −"""
    return n(v, 0) if abs(v - round(v)) < 1e-9 else n(v, 1)


def fmt_q(v):
    """照度、亮度：≥ 1 亿 / 1 万写成“x 亿 / x.x 万”，≥ 10 取整，再小保留两位有效数字"""
    if v >= 1e8:
        return ("%.1f" % (v / 1e8)).rstrip("0").rstrip(".") + " 亿"
    if v >= 1e4:
        return ("%.1f" % (v / 1e4)).rstrip("0").rstrip(".") + " 万"
    if v >= 100:
        k = 10 ** (len("%d" % int(v)) - 2)
        return "%d" % (int(round(v / k)) * k)
    if v >= 10:
        return "%d" % int(round(v))
    if v >= 1:
        return ("%.1f" % v).rstrip("0").rstrip(".")
    return "%.2g" % v


def fmt_shutter(s):
    """PPV 的 Shutter Speed 填的数（1/秒）"""
    return "%d" % int(round(s)) if s >= 10 else ("%.1f" % s).rstrip("0").rstrip(".") if s >= 1 else "%.3g" % s


def shutter_eq(s):
    return ("1/%d 秒" % int(round(s))) if s > 1 else ("%g 秒" % (1 / s))


V.update({
    "x_s16_ev": "%.2f" % s16["ev"], "x_s16_nits": "%d" % s16["nits"], "x_s16_pre": n(s16["pre"] * 1e5, 1) + "×10⁻⁵",
    "x_s16_post": "%.3f" % s16["post"], "x_s16_aces": i0(s16["aces"]), "x_s16_final": "%.3f" % s16["final"],
    "x_s16_half": i0(s16["half"]), "x_s16_double": i0(s16["double"]),
    "x_br_ev": evs(br["ev"]), "x_br_d": evs(br["d"]), "x_br_lux": i0(br["lux"]), "x_br_cd": i0(br["cd"]),
    "x_br_lm": "%d" % int(round(br["lm"], -2)),
    "x_tr_ec": n(tm["ec"]), "x_tr_sh": n(tm["shutter"]), "x_tr_sh90": n(tm["shutter_90"]), "x_tr_iso": n(tm["iso"]),
    "n_skyatmo_ship": str(SKY["SkyAtmosphereComponent"]["shipping"]), "n_fog_ship": str(SKY["ExponentialHeightFogComponent"]["shipping"]),
    "n_skylight_ship": str(SKY["SkyLightComponent"]["shipping"]), "n_ppv_ship": str(SKY["PostProcessVolume"]["shipping"]),
    "x_wr": n(0.87 / 0.18), "x_brr": n(pr["black"]["lin"] / 0.18, 2), "x_brr_inv": n(0.18 / pr["black"]["lin"]),
    "x_lava_grey": i0(0.18 * 2 ** 12), "x_lava_lo": "%d" % int(round(0.18 * 2 ** 15, -2)), "x_lava_hi": "%d" % int(round(0.18 * 2 ** 17, -2)),
})

# ACES 2.0 + sRGB 下，中灰要亮多少档才和 Filmic 的中灰一样（沿灰阶曲线插值）
_tv, _ta = D["tone"]["ev"], D["tone"]["aces_srgb"]
_mid = f["mid"]
_j = next(j for j in range(1, len(_ta)) if _ta[j] >= _mid)
V["x_grey_match_ev"] = sgn(_tv[_j - 1] + (_mid - _ta[_j - 1]) / (_ta[_j] - _ta[_j - 1]) * (_tv[_j] - _tv[_j - 1]))

# 每类场景的基准 EV100
V["h_scene_rows"] = "".join(
    '<tr><td>%s</td><td class="num">%s ~ %s</td><td class="num">%s ~ %s lux</td><td class="num">%s ~ %s</td><td class="num">%s</td>'
    '<td class="num">%s 档</td><td class="num">%s%s / %s%s</td><td>%s；%s</td></tr>'
    % (esc(s["name"]), evs(s["ev_lo"]), evs(s["ev_hi"]), fmt_q(s["lux_lo"]), fmt_q(s["lux_hi"]),
       fmt_shutter(2.0 ** s["ev_lo"] / 16), fmt_shutter(2.0 ** s["ev_hi"] / 16), "0" if s["offset"] == 0 else evs(s["offset"]),
       sgn(s["shadow"]), chip(grey_hex(s["lit"])), i0(s["lit"]), chip(grey_hex(s["dark"])), i0(s["dark"]), esc(s["key"]), esc(s["note"]))
    for s in EX["scenes"])


# EV100 对照表：每一档对应的灰卡读数、照度、快门，以及落在这一档的场景
def scene_names(e):
    hit = [re.sub(r"[（(].*?[）)]", "", s["name"]) for s in EX["scenes"] if s["ev_lo"] <= e <= s["ev_hi"]]
    if hit:
        return "<b>%s</b>" % "、".join(esc(x) for x in hit)
    ref = [r["name"] for r in EX["real_lux"] if round(r["ev_lo"]) <= e <= round(r["ev_hi"])]
    return '<span class="muted">%s</span>' % "、".join(esc(x) for x in ref) if ref else ""


V["h_ev_rows"] = "".join(
    '<tr><td class="num"><b>%s</b></td><td class="num">%s</td><td class="num">%s</td><td class="num"><b>%s</b></td><td class="num">%s</td><td>%s</td></tr>'
    % (evs(r["ev"]), fmt_q(0.18 * 2.0 ** r["ev"]), fmt_q(PI_ * 2.0 ** r["ev"]), fmt_shutter(r["s"]), shutter_eq(r["s"]), scene_names(r["ev"]))
    for r in reversed(EX["shutter_f4"]))

# EV 尺：现实世界的典型照度 + 本项目每类场景的基准
EVL_LO, EVL_HI = -7.0, 16.0


def evl_pos(v):
    return (min(max(v, EVL_LO), EVL_HI) - EVL_LO) / (EVL_HI - EVL_LO) * 100


def evl_bar(lo, hi, cls):
    if hi - lo < 0.25:
        return '<span class="evl-b dot%s" style="left:%.2f%%"></span>' % (cls, evl_pos(lo))
    return '<span class="evl-b%s" style="left:%.2f%%;width:%.2f%%"></span>' % (cls, evl_pos(lo), evl_pos(hi) - evl_pos(lo))


def evl_rng(lo, hi, f=evs):
    return f(lo) if abs(hi - lo) < 1e-9 else "%s ~ %s" % (f(lo), f(hi))


EVL_TICKS = "".join('<span style="left:%.2f%%">%s</span>' % (evl_pos(t), evs(t)) for t in (-6, -3, 0, 3, 6, 9, 12, 15))


def evl_ax(cls):
    return '<div class="evl-ax%s" aria-hidden="true">%s</div>' % (cls, EVL_TICKS)
V["h_ev_ladder"] = ('<div class="evl" role="img" aria-label="EV100 尺：现实世界从满月夜约 −6 到晴天约 15；本项目夜晚 −2 到主城白天 15">'
                    + evl_ax(" t") + '<p class="evl-g">现实世界（典型照度 → 正确曝光的 EV100）</p>'
                    + "".join('<div class="evl-r"><div class="evl-n"><b>%s</b> <span class="u">%s lux · EV100 %s</span></div><div class="evl-t">%s</div></div>'
                              % (esc(r["name"]), evl_rng(r["lo"], r["hi"], fmt_q), evl_rng(r["ev_lo"], r["ev_hi"]), evl_bar(r["ev_lo"], r["ev_hi"], ""))
                              for r in sorted(EX["real_lux"], key=lambda r: -r["ev_hi"]))
                    + '<p class="evl-g">本项目每类场景的基准 EV100（建议）</p>'
                    + "".join('<div class="evl-r"><div class="evl-n"><b>%s</b> <span class="u">EV100 %s</span></div><div class="evl-t">%s</div></div>'
                              % (esc(s["name"]), evl_rng(s["ev_lo"], s["ev_hi"]), evl_bar(s["ev_lo"], s["ev_hi"], " p")) for s in EX["scenes"])
                    + evl_ax("") + "</div>")

# 真实灯具：1 / 2 / 3 米处的照度与 EV100
V["h_lamp_rows"] = "".join(
    '<tr><td>%s</td><td class="num">%s</td>%s</tr>'
    % (esc(l["name"]), fmt_q(l["lm"]), "".join('<td class="num">%s lux<br><span class="muted">EV100 %s</span></td>' % (fmt_q(r["lux"]), n(r["ev"]))
                                                for r in l["rows"])) for l in EX["lamps"])
# 火盆：离得越远越暗
V["h_brazier"] = '<div class="dist five">' + "".join(
    '<div class="dc">%s<b>%s 米</b><span>%s lux</span><span>%s 档 · 屏幕 %s</span></div>'
    % (chip(grey_hex(r["aces"]), "chip"), n(r["d"], 1).rstrip("0").rstrip("."), fmt_q(r["lux"]), "0" if abs(r["ev"]) < 0.05 else sgn(r["ev"], 0),
       i0(r["aces"])) for r in br["rows"]) + "</div>"
# 发光体的尼特数；真实亮度参考
EMIT_NAME = {15: "晴天", 7: "室内", 4: "地牢", -1.7: "月夜"}
V["h_emit_nits_rows"] = "".join(
    '<tr><td>%s（%s）</td>%s</tr>' % (EMIT_NAME.get(e["ev"], ""), evs(e["ev"]), "".join('<td class="num">%s</td>' % fmt_q(v) for v in e["rows"]))
    for e in EX["emit_nits"])
V["h_nits_rows"] = "".join('<tr><td>%s</td><td class="num">%s</td></tr>' % (esc(nm_), fmt_q(v)) for nm_, v in EX["nits_ref"])
# 特效：旧贴图 / 新贴图
fx = EX["fx"]
V.update({"fx_i_old": n(fx["i_old"], 0), "fx_i_new": n(fx["i_new"]), "fx_de_new": n(fx["img_de"]), "fx_de_old": n(fx["img_de_old"]),
          "fx_de95": n(fx["img_de95"]), "fx_lut_de": n(fx["lut_de"]), "fx_tint_hex": fx["tint_hex"], "h_fx_tint_chip": chip(fx["tint_hex"])})
V["h_fx_stops_head"] = "".join('<th class="num">%d</th>' % st["g"] for st in fx["stops"])
V["h_fx_stops_old"] = "".join('<td class="num nw">%s%s</td>' % (chip(st["old"]), st["old"]) for st in fx["stops"])
V["h_fx_stops_new"] = "".join('<td class="num nw">%s%s</td>' % (chip(st["new"]), st["new"]) for st in fx["stops"])

# ---------------------------------------------------------------- 第三章：控制台变量（make_cvars.py）
CV = json.load(open(os.path.join(ROOT, "data", "cvars.json"), encoding="utf-8"))
CV_LEVELS = ("0", "1", "2", "3", "4")
CV_LEVEL_NAMES = ("低", "中", "高", "极高", "电影")


def cv_fmt(v):
    """变量值的显示：数字去掉多余的 0，负号用 −，布尔写成 1 / 0"""
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return "1" if s.lower() == "true" else "0"
    try:
        x = float(s.rstrip("fF"))
        s = ("%d" % x) if x == int(x) and abs(x) < 1e7 else ("%g" % x)
    except ValueError:
        pass
    return esc(s.replace("-", "−"))


def cv_default(name):
    if name not in CV["defs"]:
        raise SystemExit("cvars.json 里没有 %s，先在 make_cvars.py 的 NAMES 里加上再运行" % name)
    return cv_fmt(CV["defs"][name]["default"])


def cv_level(group, level, name):
    return CV["scalability"].get(group, {}).get(level, {}).get(name)


def cv_project(group, level, name):
    """本项目 Scalability ini 的覆盖值（Windows 平台文件优先）"""
    for f in ("Windows/WindowsScalability.ini", "DefaultScalability.ini"):
        v = CV["project_scalability"].get(f, {}).get(group, {}).get(level, {}).get(name)
        if v is not None:
            return v, f
    return None, None


def cv_matrix(group, rows):
    """一组变量在 5 个档位上的值：引擎 BaseScalability.ini，项目改过的标 ●"""
    body = []
    for name, label in rows:
        cells = []
        for lv in CV_LEVELS:
            eng = cv_level(group, lv, name)
            pv, pf = cv_project(group, lv, name)
            if pv is not None and (eng is None or cv_fmt(pv) != cv_fmt(eng)):
                cells.append('<td class="num"><span class="pj" title="本项目 %s 改为 %s">%s</span>%s</td>'
                             % (esc(pf), cv_fmt(pv), cv_fmt(pv), ('<span class="eng">（引擎 %s）</span>' % cv_fmt(eng)) if eng is not None else ""))
            else:
                cells.append('<td class="num">%s</td>' % (cv_fmt(eng) if eng is not None else '<span class="muted">—</span>'))
        body.append('<tr><td><code>%s</code><span class="cvl">%s</span></td><td class="num">%s</td>%s</tr>' % (name, esc(label), cv_default(name), "".join(cells)))
    head = '<tr><th>变量</th><th class="num">源码默认</th>%s</tr>' % "".join('<th class="num">%s</th>' % n for n in CV_LEVEL_NAMES)
    return ('<div class="scroll"><table class="t cvm"><thead>%s</thead><tbody>%s</tbody></table></div>'
            '<p class="small muted cvm-note">各档的值来自引擎 <code>BaseScalability.ini</code> 的 <code>[%s@N]</code>；“—”表示该档不设置（保持当前值）；<span class="pj"></span>是本项目 Scalability ini 改过的值（Windows 以 <code>Windows/WindowsScalability.ini</code> 为准）。</p>'
            % (head, "".join(body), group))


V["h_cvm_gi"] = cv_matrix("GlobalIlluminationQuality", [
    ("r.Lumen.DiffuseIndirect.Allow", "Lumen GI 开关"),
    ("r.Lumen.FinalGatherMethod", "最终采集：0 辐照度场 / 1 屏幕探针"),
    ("r.Lumen.ScreenProbeGather.DownsampleFactor", "屏幕探针间距（像素）"),
    ("r.Lumen.ScreenProbeGather.TracingOctahedronResolution", "每个探针的光线（边长）"),
    ("r.Lumen.ScreenProbeGather.NumAdaptiveProbes", "自适应补充探针"),
    ("r.Lumen.ScreenProbeGather.RadianceCache.ProbeResolution", "辐射缓存探针分辨率"),
    ("r.Lumen.ScreenProbeGather.RadianceCache.NumProbesToTraceBudget", "辐射缓存每帧更新探针数"),
    ("r.Lumen.ScreenProbeGather.StochasticInterpolation", "随机插值（1 更省）"),
    ("r.Lumen.ScreenProbeGather.IrradianceFormat", "辐照度格式：0 球谐 / 1 八面体"),
    ("r.Lumen.ScreenProbeGather.ShortRangeAO.DownsampleFactor", "近距离 AO 降采样"),
    ("r.Lumen.ScreenProbeGather.ShortRangeAO.BentNormal", "弯曲法线（高光遮蔽更准）"),
    ("r.LumenScene.SurfaceCache.AtlasSize", "Surface Cache 图集尺寸"),
    ("r.LumenScene.SurfaceCache.CardMaxResolution", "单张卡片最大分辨率"),
    ("r.LumenScene.DirectLighting.UpdateFactor", "直接光：每帧更新 1/N"),
    ("r.LumenScene.DirectLighting.MaxLightsPerTile", "每块缓存最多几盏灯"),
    ("r.LumenScene.Radiosity.UpdateFactor", "多次反弹：每帧更新 1/N"),
    ("r.LumenScene.Radiosity.ProbeSpacing", "多次反弹探针间距"),
    ("r.LumenScene.Radiosity.HemisphereProbeResolution", "多次反弹每探针光线（边长）"),
    ("r.Lumen.TraceMeshSDFs.Allow", "软件追踪：允许网格距离场"),
    ("r.Lumen.TranslucencyVolume.GridPixelSize", "半透明光照体积格子（像素）"),
    ("r.Lumen.HardwareRayTracing.HitLighting.Allowed", "允许命中点光照"),
    ("r.RayTracing.Scene.BuildMode", "光追场景：0 快速构建 / 1 快速追踪"),
    ("r.RayTracing.Geometry.SkeletalMeshes", "骨骼网格进光追场景"),
    ("r.SkylightIntensityMultiplier", "天光强度倍数"),
])
V["h_cvm_refl"] = cv_matrix("ReflectionQuality", [
    ("r.Lumen.Reflections.Allow", "Lumen 反射开关"),
    ("r.Lumen.Reflections.DownsampleFactor", "反射降采样"),
    ("r.Lumen.Reflections.MaxRoughnessToTraceForFoliage", "植被追反射的粗糙度上限"),
    ("r.Lumen.ScreenProbeGather.MaxRoughnessToEvaluateRoughSpecularForFoliage", "植被粗糙高光的上限"),
    ("r.Lumen.Reflections.ScreenSpaceReconstruction.NumSamples", "屏幕空间重建采样数"),
    ("r.Lumen.Reflections.ScreenSpaceReconstruction.MinWeight", "重建最小权重"),
    ("r.Lumen.TranslucencyReflections.FrontLayer.Allow", "允许半透明前层反射"),
    ("r.Lumen.TranslucencyReflections.FrontLayer.Enable", "开启半透明前层反射"),
    ("r.SSR.Quality", "屏幕空间反射质量"),
    ("r.SSR.HalfResSceneColor", "SSR 用半分辨率场景色"),
])
V["h_cvm_vsm"] = cv_matrix("ShadowQuality", [
    ("r.ShadowQuality", "阴影过滤质量（0 关闭动态阴影）"),
    ("r.Shadow.Virtual.MaxPhysicalPages", "物理页池（显存）"),
    ("r.Shadow.Virtual.ResolutionLodBiasDirectional", "平行光分辨率偏移"),
    ("r.Shadow.Virtual.ResolutionLodBiasDirectionalMoving", "平行光移动时"),
    ("r.Shadow.Virtual.ResolutionLodBiasLocal", "本地光分辨率偏移"),
    ("r.Shadow.Virtual.ResolutionLodBiasLocalMoving", "本地光移动时"),
    ("r.Shadow.Virtual.SMRT.RayCountDirectional", "平行光软阴影光线数"),
    ("r.Shadow.Virtual.SMRT.SamplesPerRayDirectional", "平行光每条光线采样"),
    ("r.Shadow.Virtual.SMRT.RayCountLocal", "本地光软阴影光线数"),
    ("r.Shadow.Virtual.SMRT.SamplesPerRayLocal", "本地光每条光线采样"),
    ("r.Shadow.Virtual.MarkCoarsePagesLocal", "本地光粗页标记"),
    ("r.Shadow.RadiusThreshold", "太小的投影物剔除（屏幕半径）"),
    ("r.DistanceFieldShadowing", "距离场阴影"),
    ("r.CapsuleShadows", "胶囊阴影"),
    ("r.LightFunctionQuality", "灯光函数质量"),
])
V["h_cvm_ml"] = cv_matrix("ShadowQuality", [
    ("r.MegaLights.NumSamplesPerPixel", "每个降采样像素的采样数"),
    ("r.MegaLights.DownsampleMode", "降采样：0 全分辨率 / 1 棋盘格 / 2 半分辨率"),
    ("r.MegaLights.ScreenTraces.Quality", "屏幕追踪：0 半分辨率 / 1 全分辨率"),
    ("r.MegaLights.FrontLayerTranslucency.Allow", "半透明前层单独计算"),
    ("r.MegaLights.Spatial.UseHistorySpatialVariance", "降噪用历史方差"),
    ("r.MegaLights.Volume.GridPixelSize", "体积网格（像素）"),
    ("r.MegaLights.Volume.GridSizeZ", "体积网格深度层数"),
])
V["h_cvm_fog"] = cv_matrix("ShadowQuality", [
    ("r.VolumetricFog", "体积雾开关"),
    ("r.VolumetricFog.GridPixelSize", "体素尺寸（像素）"),
    ("r.VolumetricFog.GridSizeZ", "深度层数"),
    ("r.VolumetricFog.HistoryMissSupersampleCount", "历史失效时的补采样"),
])
V["h_cvm_sky"] = cv_matrix("EffectsQuality", [
    ("r.SkyAtmosphere.FastSkyLUT", "天空用查找表"),
    ("r.SkyAtmosphere.FastSkyLUT.SampleCountMax", "查找表最大采样"),
    ("r.SkyAtmosphere.SampleCountMax", "逐像素最大采样"),
    ("r.SkyAtmosphere.AerialPerspectiveLUT.FastApplyOnOpaque", "不透明物体的空气透视用查找表"),
    ("r.SkyAtmosphere.AerialPerspectiveLUT.SampleCountMaxPerSlice", "空气透视每层采样"),
    ("r.SkyAtmosphere.AerialPerspectiveLUT.DepthResolution", "空气透视深度层数"),
    ("r.SkyAtmosphere.TransmittanceLUT.UseSmallFormat", "透过率表用小格式"),
])
V["h_cvm_tsr"] = cv_matrix("AntiAliasingQuality", [
    ("r.TSR.History.ScreenPercentage", "历史缓冲分辨率（%）"),
    ("r.TSR.History.UpdateQuality", "历史更新质量"),
    ("r.TSR.ShadingRejection.Flickering", "闪烁抑制"),
    ("r.TSR.RejectionAntiAliasingQuality", "历史失效处的空间抗锯齿"),
    ("r.TSR.ReprojectionField", "重投影场"),
    ("r.TSR.Resurrection", "找回被遮挡过的历史"),
])
V["h_cvm_post"] = cv_matrix("PostProcessQuality", [
    ("r.BloomQuality", "Bloom 质量"),
    ("r.Bloom.ScreenPercentage", "卷积 Bloom 分辨率（%）"),
    ("r.DepthOfFieldQuality", "景深质量"),
    ("r.MotionBlurQuality", "动态模糊质量"),
    ("r.Tonemapper.Quality", "Tonemapper 着色器组合"),
    ("r.LightShaftQuality", "光束（Light Shaft）"),
    ("r.Filter.SizeScale", "模糊滤波尺寸"),
    ("r.AmbientOcclusionLevels", "SSAO 层级（Lumen 关闭时才用）"),
    ("r.EyeAdaptationQuality", "曝光（自动 / 手动曝光的计算）"),
])


def auto_screen_percentage(display_h):
    """r.ScreenPercentage <= 0 时引擎按显示分辨率插值（LegacyScreenPercentageDriver.cpp）"""
    a = CV["auto_screen_percentage"]
    px = lambda h: h * abs(h) * 1920.0 / 1080.0
    disp = px(display_h)
    lo_d, lo_r, hi_d, hi_r = a["MinDisplayResolution"], a["MinRenderingResolution"], a["MidDisplayResolution"], a["MidRenderingResolution"]
    if disp > px(hi_d):
        lo_d, lo_r, hi_d, hi_r = a["MidDisplayResolution"], a["MidRenderingResolution"], a["MaxDisplayResolution"], a["MaxRenderingResolution"]
    t = min(max((disp - px(lo_d)) / (px(hi_d) - px(lo_d)), 0.0), 1.0)
    return (((px(lo_r) + (px(hi_r) - px(lo_r)) * t) / disp) ** 0.5) * 100


V["x_asp_1080"] = "%d" % round(auto_screen_percentage(1080))
V["x_asp_1440"] = "%d" % round(auto_screen_percentage(1440))

# 附录 A3：BaseScalability.ini 各分组所在行；全部变量的定义位置
_sl = sorted(CV["scalability_line"].items(), key=lambda kv: kv[1])
_ends = {k: (_sl[i + 1][1] - 1 if i + 1 < len(_sl) else None) for i, (k, _) in enumerate(_sl)}
_grp = {}
for k, ln in _sl:
    g = k.split("@")[0]
    _grp.setdefault(g, [ln, ln])
    _grp[g][1] = _ends[k] or ln
V["h_a3_scal"] = "、".join("<code>[%s@*]</code> %d–%d" % (g, _grp[g][0], _grp[g][1]) for g in (
    "GlobalIlluminationQuality", "ReflectionQuality", "ShadowQuality", "EffectsQuality", "AntiAliasingQuality", "PostProcessQuality"))
V["h_cv_index"] = ('<details class="cvidx"><summary>查看第三章全部 %d 个变量的源码默认值与定义位置（data/cvars.json）</summary><div class="scroll"><table class="t">'
                   '<thead><tr><th>变量</th><th class="num">默认</th><th>定义位置（Engine/Source/）</th></tr></thead><tbody>%s</tbody></table></div></details>'
                   % (len(CV["defs"]), "".join('<tr><td><code>%s</code></td><td class="num">%s</td><td><code>%s</code> %d</td></tr>'
                                              % (n, cv_fmt(d["default"]) if len(str(d["default"])) < 24 else '<span class="muted">表达式</span>', esc(d["file"]), d["line"])
                                              for n, d in sorted(CV["defs"].items()))))

payload = {
    "tone": D["tone"], "sat": D["sat"], "hue": D["hue"], "sweep_ev": D["sweep_ev"],
    "ruler": EX["ruler"], "marks": EX["marks"],
    "nonmetal": [{"name": x["name"], "lin": x["lin"], "s": x["s"]} for x in EX["nonmetal"]],
    "dE": {"cats": ["地牢场景", "选角场景", "色卡"],
           "aces_default": [sc["dungeon"]["aces_default"]["dE"], sc["selectrole"]["aces_default"]["dE"], D["checker_dE"]["aces_default"]],
           "aces_srgb": [sc["dungeon"]["aces_srgb"]["dE"], sc["selectrole"]["aces_srgb"]["dE"], D["checker_dE"]["aces_srgb"]],
           "aces_comp": [sc["dungeon"]["aces_comp"]["dE"], sc["selectrole"]["aces_comp"]["dE"], D["checker_dE"]["aces_comp"]]},
    "checker": D["checker"],
    "trans": EX["trans"],
}

tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()


def cv_token(m):
    """{{cvd:变量}} = 源码默认值；{{cvl:分组:档位:变量}} = BaseScalability.ini 里该档的值"""
    if m.group(1) == "cvd":
        return cv_default(m.group(2))
    g, lv, name = m.group(2).split(":")
    v = cv_level(g, lv, name)
    if v is None:
        raise SystemExit("BaseScalability.ini 的 [%s@%s] 里没有 %s" % (g, lv, name))
    return cv_fmt(v)


tpl = re.sub(r"\{\{(cvd|cvl):([^{}]+)\}\}", cv_token, tpl)
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(V))
if missing:
    raise SystemExit("模板里有未定义的占位符: %s" % missing)
html = re.sub(r"\{\{(\w+)\}\}", lambda m: V[m.group(1)], tpl)
html = html.replace("/*__REPORT_DATA__*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8", newline="\n").write(html)
print("index.html written, %d bytes, %d placeholders" % (len(html.encode("utf-8")), len(V)))
