$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ResearchDir = Join-Path $Root "research"
New-Item -ItemType Directory -Force -Path $ResearchDir | Out-Null

function HtmlEncode($value) {
  return [System.Net.WebUtility]::HtmlEncode([string]$value)
}

function BadgeClass($risk) {
  switch -Regex ($risk) {
    "低" { return "b-low" }
    "中" { return "b-mid" }
    "高" { return "b-high" }
    "极高|不建议" { return "b-extreme" }
    default { return "b-info" }
  }
}

$reports = @(
  [ordered]@{
    Slug = "UE5_ComputeFramework_to_UE426_Port_Assessment"
    Title = "UE5 ComputeFramework 移植到 UE4.26 评估报告"
    ShortTitle = "ComputeFramework"
    Lead = "评估 UE5 ComputeFramework 作为 GPU 计算图基础设施回迁到 UE4.26 的可行性。该模块会被 PCGCompute、DeformerGraph、Hair/Cloth 等 UE5 系统间接牵引。"
    Verdict = "不建议作为首轮公共底座完整移植。若业务只需要 PCG 或 Niagara 的最小运行时，应优先禁用对应 GPU Compute 路径；只有明确需要 GPU 数据流图时才单独立项。"
    Difficulty = "高"
    Status = "UE5 有插件，UE4.26 缺失"
    Strategy = "先做功能开关与编译剥离，后做 PC 桌面专项验证，移动端默认不承诺。"
    Kpis = @("源码规模：106 文件 / 11,484 行", "关键命中：ComputeFramework 1011、RDG 34、RenderGraph 7、TObjectPtr 26", "主要依赖：RHI、RenderCore、Renderer、Shader 参数、Editor 支持")
    Evidence = @(
      "UE5 路径：Engine\Plugins\Runtime\ComputeFramework，UE4.26 对应路径不存在。",
      "插件模块包含 Runtime ComputeFramework 与 Editor ComputeFrameworkEditor，加载期为 PostConfigInit/Default。",
      "PCG 报告中 PCGCompute 只有 9 文件 / 336 行，但真正风险来自 ComputeFramework 与 UE5 Renderer/RHI 管线。"
    )
    Modules = @(
      @("ComputeFramework Runtime", "106 文件 / 11,484 行", "RHI、RenderCore、Renderer", "高", "可移植但收益有限，需要同步处理 UE5 Shader/RDG API 差异。"),
      @("ComputeFrameworkEditor", "插件内 Editor 模块", "UnrealEd、Graph/Details、资产工具", "中高", "若不开放图编辑器，可先剥离。"),
      @("PCGCompute 依赖路径", "PCGCompute 9 文件 / 336 行", "PCG、ComputeFramework、Renderer", "高", "首轮建议禁用，保留 CPU PCG。"),
      @("Deformer/Hair/Cloth 牵引", "跨插件依赖", "Dataflow、Optimus、ChaosCloth", "极高", "不要把这些系统合并进 Compute 首轮。")
    )
    Dependencies = @("UE5 RenderGraph/RDG 调度 API", "Shader permutation 与参数元数据差异", "TObjectPtr/UE_INLINE_GENERATED_CPP_BY_NAME 降级", "移动端 RHI 能力与 Feature Level 限制")
    Platforms = @("PC：可以作为独立技术专项验证，优先 D3D11/D3D12，再看 Vulkan。", "移动端：不建议承诺 UE5 ComputeGraph 等价能力，应回退到 CPU 或预烘焙数据。")
    Plan = @("阶段 1：在 PCG/Niagara 回迁中加编译开关，默认关闭 ComputeFramework 依赖。", "阶段 2：抽取最小 Runtime，跑一个独立 ComputeGraph smoke test。", "阶段 3：按 PC 平台补齐 Shader/RDG/RHI 差异。", "阶段 4：移动端只做可用性验证，不作为功能承诺。")
  },
  [ordered]@{
    Slug = "UE5_GeometryScripting_to_UE426_Port_Assessment"
    Title = "UE5 GeometryScripting / GeometryProcessing 移植到 UE4.26 评估报告"
    ShortTitle = "GeometryScripting"
    Lead = "评估 UE5 GeometryScripting、GeometryProcessing、GeometryCore、GeometryFramework 回迁到 UE4.26 的范围。该模块直接影响 PCGGeometryScriptInterop、Biome、Water、建模工具和 DynamicMesh 工作流。"
    Verdict = "不建议全量移植 UE5 几何栈。推荐按节点/函数白名单回迁 DynamicMesh 与必要 Mesh 查询能力，PCGGeometryScriptInterop 首轮剥离。"
    Difficulty = "极高"
    Status = "UE5 有完整 Runtime 几何栈，UE4.26 仅有较旧 Experimental GeometryProcessing"
    Strategy = "建立最小几何适配层，不搬完整 GeometryScripting 蓝图库。"
    Kpis = @("GeometryScripting：213 文件 / 120,380 行", "GeometryProcessing：742 文件 / 185,761 行", "GeometryCore：317 文件 / 108,304 行", "GeometryFramework：26 文件 / 7,706 行")
    Evidence = @(
      "UE5 Runtime GeometryScripting 插件依赖 GeometryProcessing、MeshModelingToolset、PlanarCut。",
      "关键命中：GeometryScript 24466、DynamicMesh 15663、UE_INLINE_GENERATED_CPP_BY_NAME 55。",
      "PCG 报告中 PCGGeometryScriptInterop 36 文件 / 3,831 行，但其前置几何栈非常大。"
    )
    Modules = @(
      @("GeometryScriptingCore", "213 文件 / 120,380 行", "GeometryProcessing、MeshModelingToolset", "极高", "不建议整包回迁，蓝图库面太宽。"),
      @("GeometryProcessing", "742 文件 / 185,761 行", "DynamicMesh、Modeling、PlanarCut", "极高", "UE4.26 旧版本可复用一部分，需逐 API 对齐。"),
      @("GeometryCore", "317 文件 / 108,304 行", "数学/网格基础库", "高", "可抽最小 DynamicMesh 数据结构。"),
      @("GeometryFramework", "26 文件 / 7,706 行", "DynamicMeshComponent 等", "中高", "如果需要运行时几何组件再评估。"),
      @("PCGGeometryScriptInterop", "36 文件 / 3,831 行", "PCG、GeometryScripting", "极高", "首轮剥离。")
    )
    Dependencies = @("DynamicMesh 新 API", "ModelingTools/InteractiveToolsFramework 差异", "PlanarCut 与 MeshConversion", "UE5 编辑器建模模式依赖")
    Platforms = @("PC：可用于编辑器工具和离线生成，运行时需控制内存与 Cook 数据。", "移动端：不建议携带重型 GeometryScripting Runtime，优先把结果烘焙成 StaticMesh/InstancedMesh。")
    Plan = @("阶段 1：列出业务实际需要的几何节点。", "阶段 2：从 UE4.26 旧 GeometryProcessing 能覆盖的 API 开始映射。", "阶段 3：补最小 DynamicMesh/采样/布尔或投影能力。", "阶段 4：PCG Geometry Interop 单独打开，不作为 PCG 主线前置。")
  },
  [ordered]@{
    Slug = "UE5_StructUtils_PropertyBag_to_UE426_Port_Assessment"
    Title = "UE5 StructUtils / PropertyBag 移植到 UE4.26 评估报告"
    ShortTitle = "StructUtils / PropertyBag"
    Lead = "评估 UE5 动态结构、PropertyBag、StructUtilsEditor 等能力回迁到 UE4.26 的必要性。PCG 图参数、Niagara Data Channel K2 节点、StateTree/Mass 都会用到这类基础设施。"
    Verdict = "建议做最小等价层，而不是完整回迁。对 PCG/Niagara 来说，先满足参数覆盖、类型存储和编辑器 Details 即可。"
    Difficulty = "高"
    Status = "UE5 Runtime StructUtils 路径在当前源码形态不独立，StructUtilsEditor 存在；UE4.26 缺失"
    Strategy = "以 FInstancedPropertyBag 使用点为边界，构建兼容包装或降级到 UStruct/FStructOnScope。"
    Kpis = @("StructUtilsEditor：24 文件 / 4,450 行", "关键命中：PropertyBag 985、StructUtils 688、FInstancedPropertyBag 44", "PCG：PropertyBag 546 命中，StateTree/Mass：PropertyBag 346 命中")
    Evidence = @(
      "PCGGraph 依赖 StructUtils/PropertyBag.h；PCG 参数覆盖与 Graph Preconfiguration 均被牵连。",
      "Niagara Data Channel Blueprint 节点依赖 StructUtilsEditor。",
      "Mass/StateTree/SmartObjects 大量使用 PropertyBag/PropertyBinding 类型系统。"
    )
    Modules = @(
      @("Runtime PropertyBag", "跨模块基础设施", "CoreUObject、反射、序列化", "高", "建议实现兼容子集，覆盖 PCG/Niagara 必要字段类型。"),
      @("StructUtilsEditor", "24 文件 / 4,450 行", "Details、PropertyEditor、BlueprintGraph", "中高", "编辑器可降级，优先支持参数编辑。"),
      @("PCG Graph 参数", "PCG PropertyBag 546 命中", "PCG、Graph、Pin 类型", "高", "优先适配。"),
      @("StateTree/Mass 参数", "PropertyBag 346 命中", "StateTree、Mass、Binding", "高", "若不移植 AI 生态，可暂缓。")
    )
    Dependencies = @("UE5 反射与 FProperty 行为差异", "序列化版本兼容", "Details 面板自定义", "Blueprint/K2 类型暴露")
    Platforms = @("PC：主要是编辑器和 Cook 期风险，Runtime 只需稳定序列化。", "移动端：只要 Cook 后数据稳定，运行成本可控；不要保留复杂编辑器动态路径。")
    Plan = @("阶段 1：统计 PCG/Niagara 实际字段类型。", "阶段 2：实现最小 PropertyBag 数据容器和序列化。", "阶段 3：编辑器 Details 做基础编辑，不追求 UE5 完整体验。", "阶段 4：再扩展 StateTree/Mass 绑定需求。")
  },
  [ordered]@{
    Slug = "UE5_EditorFramework_to_UE426_Port_Assessment"
    Title = "UE5 编辑器框架 AssetDefinition / TypedElement 移植到 UE4.26 评估报告"
    ShortTitle = "Editor Framework"
    Lead = "评估 UE5 AssetDefinition、ContentBrowserData、TypedElementFramework、SubobjectDataInterface 等编辑器基础设施对 UE4.26 回迁的影响。PCGEditor、NiagaraEditor、MetaSoundEditor、DataflowEditor 都会触达这些框架。"
    Verdict = "不建议移植 UE5 编辑器框架本身。应把各插件的 AssetDefinition 回落到 UE4.26 的 AssetTypeActions、Content Browser 扩展和旧 Selection/Details 体系。"
    Difficulty = "高"
    Status = "UE5 多个编辑器基础模块缺失或 API 不等价"
    Strategy = "按插件逐点改造编辑器注册，而不是回迁整套 UE5 Editor Framework。"
    Kpis = @("AssetDefinition：13 文件 / 1,263 行", "ContentBrowserData：27 文件 / 7,580 行，UE4.26 有早期版本但 API 不等价", "TypedElementFramework：112 文件 / 18,169 行", "SubobjectDataInterface：13 文件 / 5,088 行")
    Evidence = @(
      "Niagara 报告指出 NiagaraEditor 依赖 AssetDefinition、ContentBrowserData、UserAssetTagsEditor 等 UE5 编辑器路径。",
      "PCGEditor 依赖 AssetDefinition、ContentBrowserData、TypedElement、SubobjectDataInterface、WidgetRegistration。",
      "Dataflow、Animation、MetaSound 等新插件也大量使用 AssetDefinition。"
    )
    Modules = @(
      @("AssetDefinition", "13 文件 / 1,263 行", "Editor 资产定义", "中", "概念简单，但牵一发动全局编辑器注册，不建议回迁框架。"),
      @("ContentBrowserData", "27 文件 / 7,580 行", "ContentBrowser、AssetRegistry", "中高", "UE4.26 有早期路径，API 不等价，按插件改造更稳。"),
      @("TypedElementFramework", "112 文件 / 18,169 行", "选择、元素句柄、编辑器工具", "高", "只为插件编辑体验不值得完整回迁。"),
      @("SubobjectDataInterface", "13 文件 / 5,088 行", "子对象编辑", "中高", "PCG/组件编辑可降级。")
    )
    Dependencies = @("UE5 ToolMenus/EditorSubsystem 差异", "Content Browser 数据源 API 差异", "TypedElement 选择模型", "资产类型动作与缩略图/菜单注册")
    Platforms = @("PC：编辑器专属，不影响运行时平台。", "移动端：Cook 后无直接影响，应避免把编辑器模块拖入 Runtime。")
    Plan = @("阶段 1：每个插件列出 AssetDefinition 类。", "阶段 2：映射到 UE4 AssetTypeActions 与 FAssetTypeActions_Base。", "阶段 3：重写菜单、缩略图、右键操作。", "阶段 4：TypedElement/Subobject 相关体验降级。")
  },
  [ordered]@{
    Slug = "UE5_Mass_StateTree_SmartObjects_to_UE426_Port_Assessment"
    Title = "UE5 Mass / StateTree / SmartObjects 移植到 UE4.26 评估报告"
    ShortTitle = "Mass / StateTree / SmartObjects"
    Lead = "评估 UE5 大规模 AI/Gameplay 框架回迁到 UE4.26 的范围。该模块族影响 PCGInstancedActorsInterop、UE5 AI 群体、SmartObject 交互和 StateTree 行为树替代路径。"
    Verdict = "不建议作为 PCG/Niagara 移植前置。若项目确实需要 UE5 群体 AI，可独立立项；PCGInstancedActorsInterop 首轮应剥离。"
    Difficulty = "极高"
    Status = "UE5 插件完整，UE4.26 缺失"
    Strategy = "业务驱动，按 MassEntity 核心、StateTree、SmartObjects 分拆，不做一次性全生态回迁。"
    Kpis = @("MassGameplay：572 文件 / 59,983 行", "MassAI：338 文件 / 33,897 行", "StateTree：488 文件 / 109,779 行", "SmartObjects：216 文件 / 38,548 行", "关键命中：StateTree 82205、SmartObject 23508、MassEntity 2639")
    Evidence = @(
      "PCGInstancedActorsInterop 依赖 InstancedActors/MassEntity，PCG 报告建议剥离。",
      "Mass/StateTree/SmartObjects 使用 PropertyBag、PropertyBinding、GameplayAbilities、TargetingSystem、WorldConditions 等多条 UE5 依赖链。",
      "源码中 WorldPartition 命中 170，说明与 UE5 世界组织系统存在交叉，需要显式剥离。"
    )
    Modules = @(
      @("MassEntity", "插件入口轻，但核心分散", "ECS Fragment/Processor", "高", "核心可研究，但生态依赖广。"),
      @("MassGameplay / MassAI", "910 文件 / 93,880 行", "Navigation、AI、Gameplay、Replication", "极高", "不是 PCG 首轮必要项。"),
      @("StateTree", "488 文件 / 109,779 行", "PropertyBinding、GameplayTasks、Editor", "极高", "独立行为框架，迁移成本高。"),
      @("SmartObjects", "216 文件 / 38,548 行", "GameplayAbilities、TargetingSystem、WorldConditions", "高", "只为 PCG 实例交互不划算。"),
      @("PCGInstancedActorsInterop", "7 文件 / 320 行", "PCG、InstancedActors、MassEntity", "极高", "首轮剥离。")
    )
    Dependencies = @("PropertyBag/StructUtils", "GameplayAbilities/TargetingSystem/WorldConditions", "UE5 Editor/AssetDefinition", "WorldPartition 相关路径需剥离")
    Platforms = @("PC：适合服务大量 NPC/实体，但需要 AI、网络、编辑器全链验证。", "移动端：Mass 可为性能服务，但回迁成本远高于收益，建议用 UE4 现有 AI/实例系统替代。")
    Plan = @("阶段 1：PCG 移植中关闭 InstancedActors/Mass Interop。", "阶段 2：若项目需要群体 AI，先做 MassEntity 核心原型。", "阶段 3：StateTree/SmartObjects 分开评估，不绑在 PCG 里。", "阶段 4：移动端只验证轻量 Processor，不引入完整生态。")
  },
  [ordered]@{
    Slug = "UE5_Water_Landmass_to_UE426_Port_Assessment"
    Title = "UE5 Water / Landmass 移植到 UE4.26 评估报告"
    ShortTitle = "Water / Landmass"
    Lead = "评估 UE5 Water、WaterAdvanced、Landmass 与 UE4.26 已有 Experimental Water/Landmass 的差异。该模块影响 Niagara Water Interop、PCGWaterInterop、地形、水体和移动端渲染。"
    Verdict = "建议保留 UE4.26 Water/Landmass 基线，只移植必要修复或接口适配，不做 UE5 Water 全量替换。PCG/Niagara Water Interop 应重写到 UE4 API 或暂缓。"
    Difficulty = "高"
    Status = "UE4.26 已有 Water/Landmass，但 UE5 API 与渲染路径差异明显"
    Strategy = "以 UE4.26 Water 为主线，UE5 只取业务需要的功能补丁。"
    Kpis = @("UE5 Water：295 文件 / 55,367 行，UE4.26 也存在", "WaterAdvanced：39 文件 / 6,354 行，UE4.26 缺失", "Landmass：46 文件 / 6,276 行，UE4.26 存在", "关键命中：RDG 230、DynamicMesh 194、TObjectPtr 237")
    Evidence = @(
      "UE5 Water 插件依赖 Landmass、Niagara、GeometryProcessing、BlueprintMaterialTextureNodes。",
      "Niagara 报告建议 Water/Niagara Water Interop 只保留 UE4.26 可用部分。",
      "PCG 报告指出 PCGWaterInterop 依赖 UE5 Water，UE4.26 Water 与 UE5 Water/Zone/Body API 不等价。"
    )
    Modules = @(
      @("Water Runtime", "295 文件 / 55,367 行", "Landmass、Niagara、GeometryProcessing", "高", "不要直接替换 UE4 Water，按 API 差异补丁。"),
      @("WaterEditor", "Water 插件内 Editor", "Landscape、Spline、Details", "中高", "编辑器工具需大量手工适配。"),
      @("WaterAdvanced", "39 文件 / 6,354 行", "Niagara、NiagaraFluids、Water", "高", "依赖 NiagaraFluids，不建议首轮。"),
      @("Landmass", "46 文件 / 6,276 行", "Landscape、Brush", "中", "UE4.26 已有，优先复用。"),
      @("PCG/Niagara Water Interop", "小插件但依赖重", "PCG、Niagara、Water", "极高", "重写或剥离。")
    )
    Dependencies = @("UE5 Water Zone/Body API", "Landscape 编辑器差异", "GeometryProcessing/DynamicMesh", "NiagaraFluids 与 RDG 渲染路径")
    Platforms = @("PC：可保留水面、浮力、Spline 工具，复杂 GPU/Fluid 后置。", "移动端：优先使用 UE4 现有水面材质和简化交互，不承诺 UE5 WaterAdvanced。")
    Plan = @("阶段 1：冻结 UE4.26 Water 基线并列 API 差异。", "阶段 2：只移植业务需要的 Water Body/材质/采样接口。", "阶段 3：Niagara/PCG Interop 走 UE4 Water 适配层。", "阶段 4：WaterAdvanced 与 Fluids 单独立项。")
  },
  [ordered]@{
    Slug = "UE5_Dataflow_ChaosCloth_to_UE426_Port_Assessment"
    Title = "UE5 Dataflow / ChaosCloth 移植到 UE4.26 评估报告"
    ShortTitle = "Dataflow / ChaosCloth"
    Lead = "评估 UE5 Dataflow、GeometryDataflow、ChaosCloth、ChaosClothAsset 与编辑器回迁到 UE4.26 的成本。该模块常被 HairStrands、Cloth、Geometry Cache、Chaos 资产管线牵引。"
    Verdict = "不建议并入 PCG/Niagara 主线。Dataflow 是 UE5 内容生产框架，ChaosClothAssetEditor 规模巨大，适合另立 DCC/角色专项。"
    Difficulty = "极高"
    Status = "UE5 插件完整，UE4.26 缺失"
    Strategy = "对现有 UE4.26 Chaos/布料能力做增量补丁，不回迁 UE5 Dataflow 图系统。"
    Kpis = @("Dataflow：367 文件 / 59,463 行", "ChaosCloth：93 文件 / 21,881 行", "ChaosClothAsset：103 文件 / 25,550 行", "ChaosClothAssetEditor：494 文件 / 70,824 行", "关键命中：Dataflow 52158、Chaos 35840、DynamicMesh 2601")
    Evidence = @(
      "Niagara 报告指出 UE5 HairStrands 依赖 Dataflow、ComputeFramework、Optimus、ChaosCaching。",
      "Dataflow.uplugin 依赖 GeometryProcessing、MeshModelingToolsetExp、ModelingToolsEditorMode、ChaosCaching、EditorDataStorage 等。",
      "ChaosClothAssetEditor 规模接近 7 万行，编辑器链路是主要风险。"
    )
    Modules = @(
      @("Dataflow Runtime/Editor", "367 文件 / 59,463 行", "GeometryProcessing、EditorDataStorage、ModelingTools", "极高", "不建议作为基础依赖回迁。"),
      @("GeometryDataflow", "13 文件 / 552 行", "Dataflow、GeometryProcessing", "高", "小但前置巨大。"),
      @("ChaosCloth Runtime", "93 文件 / 21,881 行", "Chaos、SkeletalMesh、Physics", "高", "可按角色需求单独评估。"),
      @("ChaosClothAsset", "103 文件 / 25,550 行", "Dataflow、ChaosCloth", "极高", "资产格式和 Cook 链需专项。"),
      @("ChaosClothAssetEditor", "494 文件 / 70,824 行", "AssetDefinition、Modeling、DataflowEditor", "极高", "首轮不建议。")
    )
    Dependencies = @("EditorDataStorage/AssetDefinition", "GeometryProcessing/DynamicMesh", "ChaosCaching/GeometryCache", "ModelingToolsEditorMode 与 CharacterFXEditor")
    Platforms = @("PC：可用于内容制作和高端角色效果，但需要完整编辑器链。", "移动端：运行时布料应保持 UE4 现有方案或烘焙，避免 Dataflow Runtime。")
    Plan = @("阶段 1：PCG/Niagara/Hair 回迁中剥离 Dataflow 依赖。", "阶段 2：确认角色布料是否必须升级到 UE5 ChaosCloth。", "阶段 3：若需要，只移植 Runtime 与 Cook 数据，不先做完整编辑器。", "阶段 4：DataflowEditor 后置到工具链专项。")
  },
  [ordered]@{
    Slug = "UE5_Animation_Rigging_to_UE426_Port_Assessment"
    Title = "UE5 Animation Rigging / ControlRig / IKRig 移植到 UE4.26 评估报告"
    ShortTitle = "Animation Rigging"
    Lead = "评估 UE5 ControlRig、IKRig、MotionWarping、PoseSearch 等动画工具链回迁到 UE4.26 的成本。该模块主要服务动画制作、运行时 IK、根运动修正和动作匹配。"
    Verdict = "不建议整包回迁 UE5 动画工具链。若项目需要某一能力，应按功能点移植，例如 MotionWarping Runtime 或 IKRig Retargeter，而不是把 ControlRig/RigVM 编辑器全量搬回。"
    Difficulty = "极高"
    Status = "UE5 插件完整，UE4.26 缺失或旧能力不可等价"
    Strategy = "功能点优先，运行时优先，编辑器和 RigVM 后置。"
    Kpis = @("ControlRig：1,242 文件 / 386,819 行", "IKRig：359 文件 / 92,977 行", "MotionWarping：53 文件 / 12,759 行", "PoseSearch：320 文件 / 64,883 行", "关键命中：ControlRig 73437、IKRig 16046")
    Evidence = @(
      "ControlRig 插件依赖 RigVM、PythonScriptPlugin、LevelSequenceEditor、PropertyAccessEditor、SequencerScripting 等。",
      "IKRig 依赖 ControlRig 与 FullBodyIK，插件含 Runtime、Developer、Editor 模块。",
      "MotionWarping 相对小，可作为单点回迁候选；PoseSearch/动作匹配依赖数据资产和编辑器分析链。"
    )
    Modules = @(
      @("ControlRig / RigVM", "1,242 文件 / 386,819 行", "RigVM、Sequencer、Python、Editor", "极高", "不建议整包。"),
      @("IKRig", "359 文件 / 92,977 行", "ControlRig、FullBodyIK、Editor", "极高", "如只需 Retarget，可单独裁剪。"),
      @("MotionWarping", "53 文件 / 12,759 行", "Animation、Gameplay、Root Motion", "中高", "较适合单点回迁。"),
      @("PoseSearch", "320 文件 / 64,883 行", "Animation、Editor、数据库", "高", "动作匹配专项，不绑 PCG/Niagara。")
    )
    Dependencies = @("RigVM 与 Blueprint/反射差异", "Sequencer/LevelSequenceEditor", "PropertyAccessEditor", "AssetDefinition 与动画编辑器 UI")
    Platforms = @("PC：编辑器工具价值高，但迁移面极大。", "移动端：优先保留运行时 IK/Root Motion 小功能，避免编辑器和 PoseSearch 复杂数据。")
    Plan = @("阶段 1：确认项目真正需要的动画能力。", "阶段 2：MotionWarping 可优先做 Runtime 原型。", "阶段 3：IKRig 只移植 Retarget 或求解子集。", "阶段 4：ControlRig/RigVM 全量迁移需独立长期专项。")
  },
  [ordered]@{
    Slug = "UE5_MetaSounds_Audio_to_UE426_Port_Assessment"
    Title = "UE5 MetaSounds / AudioGameplay 移植到 UE4.26 评估报告"
    ShortTitle = "MetaSounds / Audio"
    Lead = "评估 UE5 MetaSounds、MetaSoundExperimental、AudioGameplay、AudioGameplayVolume 回迁到 UE4.26 的可行性。该模块是 UE5 程序化音频图与音频 Gameplay 集成的核心。"
    Verdict = "不建议为 PCG/Niagara 迁移引入。若项目音频需求明确，可独立评估 MetaSound Runtime 子集；编辑器和资产生态迁移成本较高。"
    Difficulty = "高"
    Status = "UE5 插件完整，UE4.26 缺失；UE4.26 有部分旧音频插件但非等价"
    Strategy = "Runtime 子集先行，Editor/Graph/资产浏览器后置。"
    Kpis = @("Metasound：653 文件 / 184,483 行", "MetasoundExperimental：84 文件 / 7,311 行", "AudioGameplay：64 文件 / 6,523 行", "AudioGameplayVolume：65 文件 / 6,078 行", "关键命中：Metasound 17393、Trigger 2000、AssetDefinition 93")
    Evidence = @(
      "Metasound.uplugin 包含 GraphCore、Generator、Frontend、StandardNodes、Engine、Editor 等多个模块。",
      "插件依赖 AudioWidgets、AudioSynesthesia、ContentBrowserAssetDataSource、WaveTable。",
      "编辑器部分使用 AssetDefinition 和 ContentBrowserData，需回落到 UE4 编辑器注册体系。"
    )
    Modules = @(
      @("Metasound Runtime", "653 文件 / 184,483 行", "AudioMixer、WaveTable、Frontend", "高", "可做 Runtime 子集，但资产兼容需谨慎。"),
      @("MetasoundEditor", "UncookedOnly 模块", "AssetDefinition、GraphEditor、AudioWidgets", "高", "编辑器后置。"),
      @("MetasoundExperimental", "84 文件 / 7,311 行", "Metasound", "中高", "不进入首轮。"),
      @("AudioGameplay / Volume", "129 文件 / 12,601 行", "AudioGameplay、Volume、Gameplay", "中", "可按项目功能选择。")
    )
    Dependencies = @("AudioMixer/Quartz/WaveTable 差异", "MetaSound Frontend 序列化", "AssetDefinition/ContentBrowserData", "GraphEditor 与节点注册")
    Platforms = @("PC：Runtime 可行性取决于音频引擎差异，Editor 成本高。", "移动端：音频运行时需重点测 CPU、内存和平台后端；复杂图建议烘焙或限制节点。")
    Plan = @("阶段 1：明确是否必须支持 MetaSound 资产。", "阶段 2：只跑 Runtime 生成与基础节点。", "阶段 3：资产导入/编辑器 Graph 后置。", "阶段 4：移动端建立节点白名单。")
  },
  [ordered]@{
    Slug = "UE5_EnhancedInput_to_UE426_Port_Assessment"
    Title = "UE5 Enhanced Input 移植到 UE4.26 评估报告"
    ShortTitle = "Enhanced Input"
    Lead = "评估 UE5 Enhanced Input 回迁到 UE4.26 的成本。相比其他模块，该插件相对独立，主要提供 InputAction、InputMappingContext、Trigger、Modifier 与编辑器配置体验。"
    Verdict = "可作为中等风险的独立回迁候选。它与 PCG/Niagara 关联不强，但对项目 Gameplay 输入体系升级有直接价值。"
    Difficulty = "中"
    Status = "UE5 有插件，UE4.26 缺失"
    Strategy = "优先 Runtime，随后补 Blueprint 节点和编辑器；不依赖大型 UE5 渲染/世界系统。"
    Kpis = @("EnhancedInput：161 文件 / 40,376 行", "关键命中：EnhancedInput 975、InputAction 980、InputMappingContext 233、Trigger 1148、Modifier 289", "插件模块：EnhancedInput Runtime、InputBlueprintNodes、InputEditor")
    Evidence = @(
      "EnhancedInput.uplugin 依赖 DataValidation，模块加载期为 Runtime PreDefault、BlueprintNodes PreDefault、InputEditor Default。",
      "源码大量使用 TObjectPtr 与 UE_INLINE_GENERATED_CPP_BY_NAME，需要常规 UE5 到 UE4 降级。",
      "该模块不依赖 PCG、World Partition、DataLayer、Nanite、Lumen 等大型系统。"
    )
    Modules = @(
      @("EnhancedInput Runtime", "核心 Runtime", "Engine、InputCore、Slate 可选", "中", "首选移植对象。"),
      @("InputBlueprintNodes", "UncookedOnly", "BlueprintGraph、K2", "中", "Runtime 稳定后再做。"),
      @("InputEditor", "Editor 模块", "Details、AssetTools、DataValidation", "中", "回落到 UE4 资产/详情体系。"),
      @("Triggers/Modifiers", "输入规则库", "反射、配置资产", "中", "需要测试序列化和多人输入场景。")
    )
    Dependencies = @("TObjectPtr 降级", "Enhanced Player Input 与 UE4 PlayerInput 差异", "Blueprint 节点 API 差异", "DataValidation 可选剥离")
    Platforms = @("PC：迁移价值明确，键鼠/手柄都可验证。", "移动端：可用于触控和虚拟摇杆映射，但需要和项目现有输入层整合。")
    Plan = @("阶段 1：移植 Runtime 和基础资产类型。", "阶段 2：接入 PlayerController/LocalPlayer 输入栈。", "阶段 3：补 Blueprint 节点与编辑器。", "阶段 4：PC/移动端输入设备矩阵测试。")
  }
)

$style = @"
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --blue: #2563eb;
      --green: #15803d;
      --amber: #b45309;
      --red: #b91c1c;
      --purple: #7c3aed;
      --teal: #0f766e;
      --soft-blue: #eaf1ff;
      --soft-green: #e8f7ee;
      --soft-amber: #fff4df;
      --soft-red: #ffe7e7;
      --soft-purple: #f2eaff;
      --shadow: 0 14px 34px rgba(17, 24, 39, 0.08);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); line-height: 1.65; padding-bottom: 0; }
    header { color: #fff; background: linear-gradient(135deg, #102033 0%, #1d4ed8 52%, #0f766e 100%); padding: 42px 0 34px; }
    .wrap { width: min(1180px, calc(100% - 36px)); margin: 0 auto; }
    .title-row { display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }
    h1 { margin: 0 0 12px; font-size: clamp(28px, 4vw, 46px); line-height: 1.12; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 24px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 18px; letter-spacing: 0; }
    p { margin: 0 0 12px; }
    code { padding: 2px 6px; border-radius: 4px; background: #eef2f7; color: #172554; font-family: Consolas, "Courier New", monospace; font-size: 0.92em; }
    .subtitle { max-width: 870px; color: #dcecff; font-size: 17px; }
    .meta { display: grid; gap: 8px; min-width: 270px; padding: 14px 16px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; color: #eef8ff; font-size: 14px; }
    .toc { margin-top: 24px; display: flex; flex-wrap: wrap; gap: 10px; }
    .toc a { color: #fff; text-decoration: none; border: 1px solid rgba(255,255,255,0.28); border-radius: 999px; padding: 7px 12px; background: rgba(255,255,255,0.1); font-size: 14px; }
    main { padding: 28px 0 56px; }
    section { margin: 0 0 22px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; }
    .section-head { padding: 22px 24px 8px; }
    .section-body { padding: 0 24px 24px; }
    .grid { display: grid; gap: 16px; }
    .grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .kpi { border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fbfcff; }
    .kpi .num { font-size: 22px; line-height: 1.18; font-weight: 800; color: #111827; }
    .kpi .label { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 800; white-space: nowrap; }
    .b-low { background: var(--soft-green); color: var(--green); }
    .b-mid { background: var(--soft-amber); color: var(--amber); }
    .b-high { background: var(--soft-red); color: var(--red); }
    .b-extreme { background: #fee2ff; color: #a21caf; }
    .b-info { background: var(--soft-blue); color: var(--blue); }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 16px 24px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: #f9fbff; }
    .toolbar input, .toolbar select { min-height: 38px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0 11px; color: var(--ink); background: #fff; font-size: 14px; }
    .toolbar input { flex: 1 1 260px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 12px 10px; vertical-align: top; text-align: left; }
    th { color: #344054; background: #f8fafc; font-size: 13px; position: sticky; top: 0; z-index: 1; }
    tr:hover td { background: #fcfdff; }
    .table-scroll { overflow: auto; max-height: 650px; }
    .callout { border-left: 4px solid var(--blue); padding: 12px 14px; background: var(--soft-blue); border-radius: 6px; margin: 12px 0; }
    .callout.warn { border-left-color: var(--amber); background: var(--soft-amber); }
    .callout.danger { border-left-color: var(--red); background: var(--soft-red); }
    .source-list, .compact-list { margin: 8px 0 0; padding-left: 18px; }
    .source-list li, .compact-list li { margin: 5px 0; }
    .source-list { color: #344054; font-size: 14px; }
    .phase { display: grid; grid-template-columns: 170px 1fr; gap: 14px; padding: 15px 0; border-bottom: 1px solid var(--line); }
    .phase:last-child { border-bottom: 0; }
    .phase-title { font-weight: 800; color: #0f172a; }
    .home-link { position: fixed; top: max(16px, env(safe-area-inset-top)); right: max(16px, env(safe-area-inset-right)); z-index: 50; min-height: 42px; padding: 9px 13px; border: 1px solid rgba(255,255,255,0.55); border-radius: 999px; color: #fff; background: rgba(16,32,51,0.72); box-shadow: 0 12px 30px rgba(16,32,51,0.24); text-decoration: none; font-size: 14px; font-weight: 800; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); display: inline-flex; align-items: center; gap: 7px; }
    .home-link:hover, .home-link:focus-visible { background: rgba(15,118,110,0.92); outline: none; }
    .hidden { display: none; }
    @media (max-width: 860px) {
      .title-row, .grid.cols-2, .grid.cols-4, .phase { grid-template-columns: 1fr; }
      .section-head, .section-body, .toolbar { padding-left: 16px; padding-right: 16px; }
    }
    @media (max-width: 640px) {
      body { padding-bottom: calc(74px + env(safe-area-inset-bottom)); }
      .home-link { top: auto; right: auto; left: 50%; bottom: max(14px, env(safe-area-inset-bottom)); transform: translateX(-50%); min-height: 46px; padding: 10px 16px; color: #102033; background: rgba(255,255,255,0.94); border-color: rgba(148,163,184,0.62); box-shadow: 0 14px 36px rgba(16,32,51,0.2); }
      .home-link:hover, .home-link:focus-visible { color: #fff; }
    }
"@

foreach ($report in $reports) {
  $moduleRows = ""
  foreach ($m in $report.Modules) {
    $risk = [string]$m[3]
    $moduleRows += "<tr data-risk=""$(HtmlEncode $risk)""><td><strong>$(HtmlEncode $m[0])</strong></td><td>$(HtmlEncode $m[1])</td><td>$(HtmlEncode $m[2])</td><td><span class=""badge $(BadgeClass $risk)"">$(HtmlEncode $risk)</span></td><td>$(HtmlEncode $m[4])</td></tr>`n"
  }

  $evidenceItems = ($report.Evidence | ForEach-Object { "<li>$(HtmlEncode $_)</li>" }) -join "`n"
  $depItems = ($report.Dependencies | ForEach-Object { "<li>$(HtmlEncode $_)</li>" }) -join "`n"
  $platformItems = ($report.Platforms | ForEach-Object { "<li>$(HtmlEncode $_)</li>" }) -join "`n"
  $planRows = ""
  for ($i = 0; $i -lt $report.Plan.Count; $i++) {
    $planText = [regex]::Replace([string]$report.Plan[$i], "^阶段\s*\d+\s*[：:]\s*", "")
    $planRows += "<div class=""phase""><div class=""phase-title"">阶段 $($i + 1)</div><div>$(HtmlEncode $planText)</div></div>`n"
  }
  $kpiCards = ""
  foreach ($kpi in $report.Kpis) {
    $parts = [string]$kpi -split "：", 2
    if ($parts.Count -eq 2) {
      $kpiCards += "<div class=""kpi""><div class=""num"">$(HtmlEncode $parts[0])</div><div class=""label"">$(HtmlEncode $parts[1])</div></div>`n"
    } else {
      $kpiCards += "<div class=""kpi""><div class=""num"">证据</div><div class=""label"">$(HtmlEncode $kpi)</div></div>`n"
    }
  }

  $difficultyClass = BadgeClass $report.Difficulty
  $html = @"
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$(HtmlEncode $report.Title)</title>
  <style>
$style
  </style>
</head>
<body>
  <a class="home-link" href="../index.html" aria-label="返回报告目录">← 目录</a>
  <header>
    <div class="wrap">
      <div class="title-row">
        <div>
          <h1>$(HtmlEncode $report.Title)</h1>
          <p class="subtitle">$(HtmlEncode $report.Lead)</p>
        </div>
        <div class="meta">
          <div><strong>当前工程：</strong>UE 4.26.2</div>
          <div><strong>参考源码：</strong>UE 5.7.4</div>
          <div><strong>报告日期：</strong>2026-06-24</div>
          <div><strong>总体难度：</strong><span class="badge $difficultyClass">$(HtmlEncode $report.Difficulty)</span></div>
        </div>
      </div>
      <nav class="toc" aria-label="目录">
        <a href="#summary">结论摘要</a>
        <a href="#evidence">源码证据</a>
        <a href="#matrix">模块矩阵</a>
        <a href="#platforms">平台判断</a>
        <a href="#dependencies">依赖缺口</a>
        <a href="#plan">实施路线</a>
      </nav>
    </div>
  </header>

  <main class="wrap">
    <section id="summary">
      <div class="section-head"><h2>结论摘要</h2></div>
      <div class="section-body">
        <div class="callout warn"><strong>总判断：</strong>$(HtmlEncode $report.Verdict)</div>
        <div class="grid cols-4">
          <div class="kpi"><div class="num">$(HtmlEncode $report.ShortTitle)</div><div class="label">模块族</div></div>
          <div class="kpi"><div class="num">$(HtmlEncode $report.Status)</div><div class="label">UE5 / UE4.26 状态</div></div>
          <div class="kpi"><div class="num">$(HtmlEncode $report.Difficulty)</div><div class="label">移植难度</div></div>
          <div class="kpi"><div class="num">建议</div><div class="label">$(HtmlEncode $report.Strategy)</div></div>
        </div>
      </div>
    </section>

    <section id="evidence">
      <div class="section-head"><h2>源码证据</h2></div>
      <div class="section-body">
        <div class="grid cols-2">
          <div>
            <h3>扫描摘要</h3>
            <div class="grid cols-2">
              $kpiCards
            </div>
          </div>
          <div>
            <h3>关键证据</h3>
            <ul class="source-list">
              $evidenceItems
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section id="matrix">
      <div class="section-head"><h2>模块难度矩阵</h2></div>
      <div class="toolbar">
        <input id="moduleSearch" type="search" placeholder="搜索模块、依赖、建议">
        <select id="riskFilter">
          <option value="">全部难度</option>
          <option value="中">中</option>
          <option value="中高">中高</option>
          <option value="高">高</option>
          <option value="极高">极高</option>
        </select>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>模块/能力</th><th>规模</th><th>主要依赖</th><th>风险</th><th>移植建议</th></tr>
          </thead>
          <tbody id="moduleRows">
            $moduleRows
          </tbody>
        </table>
      </div>
    </section>

    <section id="platforms">
      <div class="section-head"><h2>移动端与 PC 判断</h2></div>
      <div class="section-body">
        <ul class="compact-list">
          $platformItems
        </ul>
      </div>
    </section>

    <section id="dependencies">
      <div class="section-head"><h2>主要依赖缺口</h2></div>
      <div class="section-body">
        <p>本节只列本模块族迁移时最容易放大工作量的依赖。Lumen、Nanite、World Partition、DataLayer 不作为本批报告目标；遇到相关调用点时默认剥离或降级。</p>
        <ul class="compact-list">
          $depItems
        </ul>
      </div>
    </section>

    <section id="plan">
      <div class="section-head"><h2>建议实施路线</h2></div>
      <div class="section-body">
        $planRows
      </div>
    </section>
  </main>

  <script>
    const search = document.getElementById("moduleSearch");
    const risk = document.getElementById("riskFilter");
    const rows = Array.from(document.querySelectorAll("#moduleRows tr"));
    function applyFilters() {
      const term = search.value.trim().toLowerCase();
      const selectedRisk = risk.value;
      rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        const rowRisk = row.dataset.risk || "";
        const showByText = !term || text.includes(term);
        const showByRisk = !selectedRisk || rowRisk === selectedRisk;
        row.classList.toggle("hidden", !(showByText && showByRisk));
      });
    }
    search.addEventListener("input", applyFilters);
    risk.addEventListener("change", applyFilters);
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener("click", event => {
        const target = document.querySelector(anchor.getAttribute("href"));
        if (target) {
          event.preventDefault();
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  </script>
</body>
</html>
"@

  $path = Join-Path $ResearchDir "$($report.Slug).html"
  Set-Content -LiteralPath $path -Value $html -Encoding UTF8
}

Write-Output "Generated $($reports.Count) module reports in $ResearchDir"
