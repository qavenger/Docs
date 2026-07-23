cmake_minimum_required(VERSION 3.24)
project(DX12_GI_Lab LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "DX12_GI_Lab only builds on Windows.")
endif()

add_executable(DX12_GI_Lab
  src/main.cpp
  src/PrtScene.cpp
  src/PrtScene.h
  shaders/PrtCompute.hlsl
  shaders/Scene.hlsl
)

# These files are compiled at runtime by CompileShader so shader diagnostics stay
# visible to learners after editing.  Marking them as header-only prevents the
# Visual Studio generator from also invoking FXC with its default `main` entry.
set_source_files_properties(
  shaders/PrtCompute.hlsl
  shaders/Scene.hlsl
  PROPERTIES HEADER_FILE_ONLY TRUE
)

target_compile_features(DX12_GI_Lab PRIVATE cxx_std_20)
target_compile_definitions(DX12_GI_Lab PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(DX12_GI_Lab PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
target_link_libraries(DX12_GI_Lab PRIVATE d3d12 dxgi d3dcompiler)

set_target_properties(DX12_GI_Lab PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:DX12_GI_Lab>"
)

add_custom_command(TARGET DX12_GI_Lab POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:DX12_GI_Lab>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/PrtCompute.hlsl"
          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Scene.hlsl"
          "$<TARGET_FILE_DIR:DX12_GI_Lab>/shaders"
  COMMENT "Copying inspectable HLSL next to the executable"
)
