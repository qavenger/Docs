本目录保存由原生 RenderGraph 样例实际生成的预览图，以及可随文档拷贝的源码快照。

- native_voxel_gi.bmp：08_gi_workbench --validate 的 D3D12 frame 0 GPU 读回
- native_voxel_gi_indirect.bmp：同一次验证的 indirect-only GPU 读回
- rendergraph_gi_source_snapshot.zip：本次 GI 扩展新增/修改的 C++ / HLSL /
  manifest / 顶层 CMake 入口。它需要叠加到兼容的 RenderGraph 工程；ZIP 未包含
  engine/ 与 third_party/，不能作为完整独立工程构建。
