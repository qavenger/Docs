# -*- coding: utf-8 -*-
"""把 data/*.json 注入 report_template.html，生成 ../index.html。
模板里的 {{name}} 由下面的 V 字典替换（数字全部从数据算出来，避免手抄出错）；
/*__REPORT_DATA__*/ 替换成图表用的 JSON。运行：python -I build_html.py"""
import json, os, re

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

payload = {
    "tone": D["tone"], "sat": D["sat"], "hue": D["hue"], "sweep_ev": D["sweep_ev"],
    "dE": {"cats": ["地牢场景", "选角场景", "色卡"],
           "aces_default": [sc["dungeon"]["aces_default"]["dE"], sc["selectrole"]["aces_default"]["dE"], D["checker_dE"]["aces_default"]],
           "aces_srgb": [sc["dungeon"]["aces_srgb"]["dE"], sc["selectrole"]["aces_srgb"]["dE"], D["checker_dE"]["aces_srgb"]],
           "aces_comp": [sc["dungeon"]["aces_comp"]["dE"], sc["selectrole"]["aces_comp"]["dE"], D["checker_dE"]["aces_comp"]]},
    "checker": D["checker"],
}

tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(V))
if missing:
    raise SystemExit("模板里有未定义的占位符: %s" % missing)
html = re.sub(r"\{\{(\w+)\}\}", lambda m: V[m.group(1)], tpl)
html = html.replace("/*__REPORT_DATA__*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8", newline="\n").write(html)
print("index.html written, %d bytes, %d placeholders" % (len(html.encode("utf-8")), len(V)))
