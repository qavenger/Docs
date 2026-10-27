本目录保存由原生 RenderGraph 样例实际生成的预览图，以及可随文档拷贝的源码快照。

预览图（均为 08_gi_workbench --validate --capture-bmp=… 的 D3D12 frame GPU 读回，
320×180，*_indirect 为 indirect-only 视图、天空渲黑）：

- native_gi_voxel.bmp / native_gi_voxel_indirect.bmp   Voxel Cone Tracing
- native_gi_rsm.bmp / native_gi_rsm_indirect.bmp       Reflective Shadow Maps
- native_gi_ssgi.bmp / native_gi_ssgi_indirect.bmp     Screen-Space GI
- native_gi_probe.bmp / native_gi_probe_indirect.bmp   SH Irradiance Probes
- native_gi_rcgi.bmp / native_gi_rcgi_indirect.bmp     Radiance Cascades（flatland 俯视图；
                                                       _indirect 为 fluence-only）
- native_gi_ddgi.bmp / native_gi_ddgi_indirect.bmp     DDGI teaching-lite（octahedral atlas
                                                       + Chebyshev + hysteresis）
- native_gi_prt.bmp / native_gi_prt_indirect.bmp       PRT transfer probes（环境重照明）

- rendergraph_gi_source_snapshot.zip：GI 工作台全部 C++ / HLSL / manifest /
  顶层 CMake 入口（七方案：voxel/rsm/ssgi/probe/rcgi/ddgi/prt）。它需要叠加到
  兼容的 RenderGraph 工程；ZIP 未包含 engine/ 与 third_party/，不能作为完整
  独立工程构建。
