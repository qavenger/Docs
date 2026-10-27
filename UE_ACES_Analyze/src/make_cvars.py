# -*- coding: utf-8 -*-
"""第三章（效果与性能：控制台变量）用到的数据：data/cvars.json。
从引擎源码读出每个变量的默认值、说明和定义位置；从引擎 BaseScalability.ini 读出各画质档位的值；
再读本项目的 Scalability / Engine ini，标出项目改过的地方。build_html.py 用它生成第三章的表格。

需要本机有引擎源码和项目（路径可以用环境变量覆盖）：
  UE_ENGINE_DIR   默认 E:\\TorchLight_UE5\\Engine\\Engine
  UE_PROJECT_DIR  默认 E:\\TorchLight_UE5\\Torchlight_Migrate
运行：python -I make_cvars.py"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
ENGINE = os.environ.get("UE_ENGINE_DIR", r"E:\TorchLight_UE5\Engine\Engine")
PROJECT = os.environ.get("UE_PROJECT_DIR", r"E:\TorchLight_UE5\Torchlight_Migrate")
SCAN_DIRS = ["Source/Runtime/Renderer", "Source/Runtime/Engine", "Source/Runtime/RenderCore", "Source/Runtime/Core/Private/HAL"]

# 第三章里出现的变量
NAMES = """
r.Lumen.DiffuseIndirect.Allow r.Lumen.FinalGatherMethod r.Lumen.ScreenProbeGather.DownsampleFactor
r.Lumen.ScreenProbeGather.TracingOctahedronResolution r.Lumen.ScreenProbeGather.NumAdaptiveProbes
r.Lumen.ScreenProbeGather.RadianceCache.ProbeResolution r.Lumen.ScreenProbeGather.RadianceCache.NumProbesToTraceBudget
r.Lumen.ScreenProbeGather.RadianceCache.GridResolution r.Lumen.ScreenProbeGather.StochasticInterpolation
r.Lumen.ScreenProbeGather.IrradianceFormat r.Lumen.ScreenProbeGather.ShortRangeAO r.Lumen.ScreenProbeGather.ShortRangeAO.DownsampleFactor
r.Lumen.ScreenProbeGather.ShortRangeAO.BentNormal r.Lumen.ScreenProbeGather.ShortRangeAO.HorizonSearch.SliceCount
r.Lumen.ScreenProbeGather.IntegrateDownsampleFactor r.Lumen.ScreenProbeGather.Temporal.MaxFramesAccumulated
r.Lumen.ScreenProbeGather.MaxRayIntensity r.Lumen.ScreenProbeGather.FullResolutionJitterWidth
r.LumenScene.SurfaceCache.AtlasSize r.LumenScene.SurfaceCache.CardTexelDensityScale r.LumenScene.SurfaceCache.CardMaxResolution
r.LumenScene.SurfaceCache.CardMinResolution r.LumenScene.DirectLighting.UpdateFactor r.LumenScene.DirectLighting.MaxLightsPerTile
r.LumenScene.Radiosity.UpdateFactor r.LumenScene.Radiosity.ProbeSpacing r.LumenScene.Radiosity.HemisphereProbeResolution
r.Lumen.TraceMeshSDFs r.Lumen.TraceMeshSDFs.Allow r.Lumen.TranslucencyVolume.GridPixelSize r.Lumen.TranslucencyVolume.TraceFromVolume
r.Lumen.TranslucencyVolume.RadianceCache.NumProbesToTraceBudget r.Lumen.HardwareRayTracing r.Lumen.HardwareRayTracing.LightingMode
r.Lumen.HardwareRayTracing.HitLighting.Allowed r.Lumen.HardwareRayTracing.MaxIterations r.Lumen.AsyncCompute r.Lumen.DiffuseIndirect.SSAO
r.Lumen.Visualize r.LumenScene.FarField r.RayTracing.Culling r.RayTracing.Culling.Radius r.RayTracing.Culling.Angle
r.RayTracing.Scene.BuildMode r.RayTracing.Geometry.SkeletalMeshes r.MeshCardRepresentation.SkeletalMesh
r.RayTracing.DynamicGeometry.MaxUpdatePrimitivesPerFrame r.SkylightIntensityMultiplier r.DistanceFieldAO r.AOQuality
r.GenerateMeshDistanceFields r.EyeAdaptation.CachedLightingPreExposure
r.Lumen.Reflections.Allow r.Lumen.Reflections.DownsampleFactor r.Lumen.Reflections.DownsampleCheckerboard
r.Lumen.Reflections.MaxRoughnessToTrace r.Lumen.Reflections.MaxRoughnessToTraceClamp r.Lumen.Reflections.MaxRoughnessToTraceForFoliage
r.Lumen.ScreenProbeGather.MaxRoughnessToEvaluateRoughSpecularForFoliage r.Lumen.Reflections.ScreenSpaceReconstruction.NumSamples
r.Lumen.Reflections.ScreenSpaceReconstruction.MinWeight r.Lumen.Reflections.RadianceCache r.Lumen.TranslucencyReflections.FrontLayer.Allow
r.Lumen.TranslucencyReflections.FrontLayer.Enable r.SSR.Quality r.SSR.HalfResSceneColor
r.Shadow.Virtual.MaxPhysicalPages r.Shadow.Virtual.ResolutionLodBiasDirectional r.Shadow.Virtual.ResolutionLodBiasDirectionalMoving
r.Shadow.Virtual.ResolutionLodBiasLocal r.Shadow.Virtual.ResolutionLodBiasLocalMoving r.Shadow.Virtual.SMRT.RayCountDirectional
r.Shadow.Virtual.SMRT.SamplesPerRayDirectional r.Shadow.Virtual.SMRT.RayCountLocal r.Shadow.Virtual.SMRT.SamplesPerRayLocal
r.Shadow.Virtual.MarkCoarsePagesLocal r.Shadow.Virtual.Clipmap.FirstLevel r.Shadow.Virtual.Clipmap.LastLevel r.Shadow.Virtual.Cache
r.Shadow.Virtual.MarkPixelPagesMipModeLocal r.Shadow.Virtual.NormalBias r.Shadow.RadiusThreshold r.DistanceFieldShadowing r.CapsuleShadows
r.ContactShadows r.LightFunctionQuality r.ShadowQuality
r.MegaLights.EnableForProject r.MegaLights.Allowed r.MegaLights.NumSamplesPerPixel r.MegaLights.DownsampleMode
r.MegaLights.ScreenTraces.Quality r.MegaLights.FrontLayerTranslucency.Allow r.MegaLights.Spatial.UseHistorySpatialVariance
r.MegaLights.Volume r.MegaLights.Volume.GridPixelSize r.MegaLights.Volume.GridSizeZ r.MegaLights.DirectionalLights
r.MegaLights.LightAttenuationFalloff r.MegaLights.HardwareRayTracing r.Visibility.LocalLightPrimitiveInteraction r.MegaLights.Visualize
r.SkyAtmosphere.FastSkyLUT r.SkyAtmosphere.FastSkyLUT.Width r.SkyAtmosphere.FastSkyLUT.Height r.SkyAtmosphere.FastSkyLUT.SampleCountMax
r.SkyAtmosphere.SampleCountMax r.SkyAtmosphere.AerialPerspectiveLUT.FastApplyOnOpaque r.SkyAtmosphere.AerialPerspectiveLUT.SampleCountMaxPerSlice
r.SkyAtmosphere.AerialPerspectiveLUT.DepthResolution r.SkyAtmosphere.AerialPerspectiveLUT.Depth r.SkyAtmosphere.TransmittanceLUT.UseSmallFormat
r.SkyLight.RealTimeReflectionCapture.TimeSlice r.SkyLight.RealTimeReflectionCapture.TimeSlice.SkyCloudCubeFacePerFrame
r.SkyLight.RealTimeReflectionCapture.ShadowFromOpaque r.VolumetricRenderTarget r.VolumetricRenderTarget.Mode
r.VolumetricCloud.ViewRaySampleMaxCount r.VolumetricCloud.ReflectionRaySampleMaxCount r.VolumetricCloud.ShadowMap.MaxResolution
r.VolumetricFog r.VolumetricFog.GridPixelSize r.VolumetricFog.GridSizeZ r.VolumetricFog.HistoryMissSupersampleCount r.VolumetricFog.HistoryWeight
r.LocalFogVolume.RenderIntoVolumetricFog r.LocalFogVolume.GlobalStartDistance r.LocalFogVolume.ApplyOnTranslucent
r.ScreenPercentage r.AntiAliasingMethod r.TSR.History.ScreenPercentage r.TSR.History.UpdateQuality r.TSR.ShadingRejection.Flickering
r.TSR.RejectionAntiAliasingQuality r.TSR.ReprojectionField r.TSR.Resurrection r.DynamicRes.OperationMode r.DynamicRes.FrameTimeBudget
r.BloomQuality r.Bloom.ScreenPercentage r.DepthOfFieldQuality r.MotionBlurQuality r.Tonemapper.Quality r.LightShaftQuality r.Filter.SizeScale
r.AmbientOcclusionLevels r.EyeAdaptationQuality r.RDG.AsyncCompute
""".split()

GROUPS = ("AntiAliasingQuality", "ViewDistanceQuality", "ShadowQuality", "GlobalIlluminationQuality", "ReflectionQuality",
          "PostProcessQuality", "TextureQuality", "EffectsQuality", "FoliageQuality", "ShadingQuality")


def split_args(s):
    """按顶层逗号切分参数（考虑括号与字符串）"""
    out, depth, cur, i, instr = [], 0, [], 0, False
    while i < len(s):
        c = s[i]
        if instr:
            cur.append(c)
            if c == "\\":
                cur.append(s[i + 1]); i += 2; continue
            if c == '"':
                instr = False
        elif c == '"':
            instr = True; cur.append(c)
        elif c in "([{":
            depth += 1; cur.append(c)
        elif c in ")]}":
            depth -= 1; cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        out.append("".join(cur).strip())
    return out


def text_of(arg):
    parts = re.findall(r'TEXT\(\s*"((?:[^"\\]|\\.)*)"\s*\)|"((?:[^"\\]|\\.)*)"', arg)
    return "".join(a or b for a, b in parts).replace("\\n", "\n").replace('\\"', '"').replace("\\t", " ").strip()


def clean_default(v):
    v = v.strip()
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)f?", v)
    if m:
        x = float(m.group(1))
        return ("%d" % x) if x == int(x) and abs(x) < 1e7 else ("%g" % x)
    return v


DECL = re.compile(r"(TAutoConsoleVariable\s*<\s*([\w:]+)\s*>|FAutoConsoleVariableRef|FAutoConsoleVariable)\s+(\w+)\s*\(\s*TEXT\(\s*\"([^\"]+)\"\s*\)")


def scan_defs(wanted):
    defs = {}
    for sub in SCAN_DIRS:
        for dp, _, fs in os.walk(os.path.join(ENGINE, sub)):
            for fn in fs:
                if not fn.endswith((".cpp", ".h")):
                    continue
                p = os.path.join(dp, fn)
                src = open(p, encoding="utf-8", errors="replace").read()
                if "ConsoleVariable" not in src:
                    continue
                for m in DECL.finditer(src):
                    name = m.group(4)
                    if name not in wanted or name in defs:
                        continue
                    start = src.index("(", m.end(3))
                    depth, j, instr = 0, start, False
                    while j < len(src):
                        c = src[j]
                        if instr:
                            if c == "\\":
                                j += 2; continue
                            if c == '"':
                                instr = False
                        elif c == '"':
                            instr = True
                        elif c == "(":
                            depth += 1
                        elif c == ")":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    args = split_args(src[start + 1:j])
                    default, vtype = args[1] if len(args) > 1 else "", m.group(2) or ""
                    if "Ref" in m.group(1):
                        vm = re.search(r"\b(int32|float|bool|int|uint32|double)\s+" + re.escape(default) + r"\s*=\s*([^;]+);", src)
                        vtype, default = (vm.group(1), vm.group(2)) if vm else ("", "")
                    defs[name] = {"default": clean_default(default), "type": vtype,
                                  "file": os.path.relpath(p, os.path.join(ENGINE, "Source")).replace("\\", "/"),
                                  "line": src.count("\n", 0, m.start(4)) + 1,
                                  "help": text_of(",".join(args[2:-1])) if len(args) > 3 else text_of(args[2]) if len(args) > 2 else ""}
    return defs


def parse_ini(path):
    """返回 {section: {key: value}}；忽略注释，“-key” 视为删除"""
    sec, out = None, {}
    if not os.path.exists(path):
        return out
    for raw in open(path, encoding="utf-8-sig", errors="replace"):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]; out.setdefault(sec, {}); continue
        if sec is None or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("-"):
            continue
        k = k.lstrip("+")
        v = v.split(";", 1)[0].strip()
        out[sec][k] = v
    return out


def scalability(ini):
    res = {}
    for sec, kv in ini.items():
        m = re.fullmatch(r"(\w+)@(\d|Cine)", sec)
        if not m or m.group(1) not in GROUPS:
            continue
        lvl = "4" if m.group(2) == "Cine" else m.group(2)
        res.setdefault(m.group(1), {})[lvl] = {k: v for k, v in kv.items() if k.lower() in WANTED_LOWER}
    return res


WANTED_LOWER = {n.lower(): n for n in NAMES}
defs = scan_defs(set(NAMES))
base = parse_ini(os.path.join(ENGINE, "Config", "BaseScalability.ini"))
proj_files = {"DefaultScalability.ini": os.path.join(PROJECT, "Config", "DefaultScalability.ini"),
              "Windows/WindowsScalability.ini": os.path.join(PROJECT, "Config", "Windows", "WindowsScalability.ini")}
eng_files = {"DefaultEngine.ini": os.path.join(PROJECT, "Config", "DefaultEngine.ini"),
             "Windows/WindowsEngine.ini": os.path.join(PROJECT, "Config", "Windows", "WindowsEngine.ini")}


def canon(kv):
    return {WANTED_LOWER[k.lower()]: v for k, v in kv.items() if k.lower() in WANTED_LOWER}


out = {
    "engine": "5.8.4", "date": "2026-10-27",
    "defs": defs,
    "missing": sorted(set(NAMES) - set(defs)),
    "scalability": {g: {l: canon(kv) for l, kv in lv.items()} for g, lv in scalability(base).items()},
    "scalability_line": {},
    "project_scalability": {f: {g: {l: canon(kv) for l, kv in lv.items()} for g, lv in scalability(parse_ini(p)).items()} for f, p in proj_files.items()},
    "project_engine": {f: {s: canon(kv) for s, kv in parse_ini(p).items() if s in ("/Script/Engine.RendererSettings", "SystemSettings", "ConsoleVariables") and canon(kv)}
                       for f, p in eng_files.items()},
}
# 自动渲染分辨率（r.ScreenPercentage <= 0 时按显示分辨率插值）的分段点
out["auto_screen_percentage"] = {k: float(v) for k, v in parse_ini(os.path.join(ENGINE, "Config", "BaseEngine.ini")).get("Rendering.AutoScreenPercentage", {}).items()}
# BaseScalability.ini 里每个分组小节的起始行号（附录引用）
for i, line in enumerate(open(os.path.join(ENGINE, "Config", "BaseScalability.ini"), encoding="utf-8", errors="replace"), 1):
    m = re.match(r"\[(\w+)@(\d|Cine)\]", line)
    if m and m.group(1) in GROUPS:
        out["scalability_line"]["%s@%s" % (m.group(1), m.group(2))] = i
json.dump(out, open(os.path.join(ROOT, "data", "cvars.json"), "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print("defs %d / %d, missing: %s" % (len(defs), len(NAMES), out["missing"]))
