# DX12 Radiance Transfer Probe Lab

这是 HTML 版 PRT 3D Demo 的原生 Direct3D 12 对照实现。两边使用同一组教学数据与公式：

1. `ShadeWallBricks`：当前动态灯重新照亮 8×5 个墙面 bricks，得到 `R[brick].rgb`。
2. `BuildTransferMatrix`：固定几何下生成 120×40 的 `W[receiver, brick]`。教学程序在启动时执行这一步，以便单步调试；生产项目通常离线烘焙并序列化该矩阵。
3. `PrtCompute.hlsl`：DirectCompute 执行 `E = W · R`。
4. 左半屏读取上次保存的 `bakedE`；右半屏读取当前 compute 输出 `E`。

量纲约定与 HTML 页一致：`R = albedo/π × incident` 是墙面出射 radiance，`W = visibility × cos_receiver × cos_brick × area / distance²` 是 radiance→irradiance 的几何项，因此 `W` 不会再重复除以 π。

这里的“一跳”是教学化的 diffuse brick→receiver 传输，接近《The Division》Radiance Transfer Probes 的数据流。生产实现会使用稀疏连接、bricks/sector streaming、visibility 压缩和更复杂的 relighting。

交互式公式、逐步代值和多行源码对照见相对路径入口
[`index.html`](index.html)。若要继续学习多 Pass 资源依赖、自动 barrier、Texture3D
与 D3D12/Vulkan 共用 HLSL，请打开
[`../RenderGraph_GI_Lab/index.html`](../RenderGraph_GI_Lab/index.html)。

## 构建

在仓库目录执行：

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

也可使用 Visual Studio 2022：

```powershell
cmake -S . -B build-vs2022 -G "Visual Studio 17 2022" -A x64
cmake --build build-vs2022 --config Release
```

工程只依赖 Windows SDK 自带的 `d3d12`、`dxgi`、`d3dcompiler` 和 DirectXMath，不依赖 NuGet 或第三方渲染框架。HLSL 在启动时通过 `D3DCompileFromFile` 编译，修改 shader 后无需重新配置 CMake。

## 运行

```powershell
build\Release\DX12_GI_Lab.exe
```

控制：

- `A / D` 或左右方向键：移动动态灯。
- `Space`：自动旋转灯。
- `R`：把当前 PRT 输出复制为左侧 baked irradiance。
- 鼠标左键拖动：旋转相机。
- 鼠标滚轮：缩放。
- `Esc`：退出。

窗口左侧是旧 Irradiance Probe 答案，右侧是本帧 `W·R`。移动灯后，右侧同帧变化，左侧保持不动；按 `R` 后二者重新一致。动态灯沿靠近彩色后墙的弧线移动，因此可以先观察墙面 brick 的直接受光变化，再观察地面 receiver 的间接光如何跟随变化。

普通交互运行会主动脱离控制台，只显示 DX12 教学窗口；`--self-test` 会保留控制台，便于 CI 读取结果与退出码。

## 无窗口自检

```powershell
build\Release\DX12_GI_Lab.exe --self-test
```

自检会：

1. 创建硬件 D3D12 device，失败时回退 WARP。
2. 用 HLSL compute 计算 120 个 receiver。
3. readback GPU 输出并与 `EvaluateTransferCpu` 对比。
4. 最大绝对误差不超过 `1e-5` 时返回 0，否则返回非零。
5. Debug 配置还会读取 D3D12 Info Queue；出现 ERROR/CORRUPTION 时自检失败并打印真实 debug-layer 消息。

本工程在 2026-07-22 使用 Visual Studio 18 2026 x64 验证：

```text
adapter=NVIDIA GeForce RTX 5060 Ti backend=hardware
receivers=120 bricks=40 max_abs_error=2.98023e-08
d3d12_debug_layer=enabled stored_messages=0 warnings=0 errors=0
SELF_TEST_PASSED
```

这里记录的是一次可复核的构建快照，并非对所有 GPU 的硬编码预期；你的适配器名称和浮点末位可能不同，但阈值仍为 `1e-5`。

## 函数都在哪里

| 文档名 | 真实定义 | 作用 |
|---|---|---|
| `SegmentIntersectsAabb` | `src/PrtScene.cpp` | slab line-segment/AABB visibility test |
| `BuildTransferMatrix` | `src/PrtScene.cpp` | 计算固定 form-factor 近似矩阵 W |
| `ShadeWallBricks` | `src/PrtScene.cpp` | 当前灯 → brick radiance R |
| `EvaluateTransferCpu` | `src/PrtScene.cpp` | CPU 参考 `E=W·R`，供自检 |
| `CSMain` | `shaders/PrtCompute.hlsl` | GPU `E=W·R` |
| `CompileShader` | `src/main.cpp` | 调用 Windows SDK `D3DCompileFromFile` 并保留完整错误日志 |
| `InspectDebugMessages` | `src/main.cpp` | Debug 自检读取 D3D12 Info Queue 并拒绝错误消息 |

没有仅为伪代码出现的 helper。`main.cpp` 中的 DX12 辅助函数也都有定义；它们是 API 基础设施，不是 PRT 算法的一部分。

## 为什么没有把它包装成“高性能模板”

为了让初学者能单步看到 compute 输出，示例每帧会把 `E` readback 到 CPU，再用它更新教学用顶点颜色。这会产生同步等待，不能当作生产性能范例。真实引擎应让 graphics shader 直接读取 GPU `E` buffer，或把结果写入 probe atlas，避免 readback。

## 目录与便携版

- `index.html`：离线项目入口，所有源码链接均为相对路径。
- `src/`：CPU 参考、共享场景/PRT 算法及 D3D12 主程序。
- `shaders/`：可直接修改并在启动时编译的 HLSL。
- `bin/Release/`：已验证的便携 Release；必须保留其相邻的 `shaders/` 子目录。
- `assets/`：本次硬件运行截图。
- `../RenderGraph_GI_Lab/`：六 Pass 原生 Voxel GI、GPU 回读截图与可携带源码快照。

便携版可直接运行：

```powershell
bin\Release\DX12_GI_Lab.exe
bin\Release\DX12_GI_Lab.exe --self-test
```

## 对应资料

- Microsoft Direct3D 12 基础组件：<https://learn.microsoft.com/windows/win32/direct3d12/creating-a-basic-direct3d-12-component>
- Microsoft DirectX Graphics Samples：<https://github.com/microsoft/DirectX-Graphics-Samples>
- `ID3D12GraphicsCommandList::Dispatch`：<https://learn.microsoft.com/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-dispatch>
- UAV barrier：<https://learn.microsoft.com/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_uav_barrier>
- WebGPU 对 D3D12/Vulkan/Metal 的映射目标：<https://gpuweb.github.io/gpuweb/explainer/>
