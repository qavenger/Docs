#include "PrtScene.h"

#include <windows.h>
#include <windowsx.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace
{
constexpr UINT kFrameCount = 2;
constexpr UINT kWindowWidth = 1280;
constexpr UINT kWindowHeight = 720;
constexpr DXGI_FORMAT kBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

void CheckHr(HRESULT result, const char* expression)
{
    if (SUCCEEDED(result))
        return;
    std::ostringstream message;
    message << expression << " failed with HRESULT 0x" << std::hex
            << static_cast<unsigned long>(result);
    throw std::runtime_error(message.str());
}

#define DX_CHECK(expression) CheckHr((expression), #expression)

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
{
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC BufferDescription(
    UINT64 size,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Alignment = 0;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc = {1, 0};
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    return description;
}

D3D12_RESOURCE_BARRIER TransitionBarrier(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

struct DebugMessageSummary
{
    UINT64 stored = 0;
    UINT64 warnings = 0;
    UINT64 errors = 0;
};

DebugMessageSummary InspectDebugMessages(ID3D12Device* device)
{
    DebugMessageSummary summary{};
#if defined(_DEBUG)
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue))))
        return summary;

    summary.stored = queue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 index = 0; index < summary.stored; ++index)
    {
        SIZE_T byteCount = 0;
        queue->GetMessage(index, nullptr, &byteCount);
        if (byteCount == 0)
            continue;
        std::vector<std::byte> storage(byteCount);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(queue->GetMessage(index, message, &byteCount)))
            continue;
        if (message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
        {
            ++summary.warnings;
            std::cerr << "D3D12_DEBUG_WARNING id=" << message->ID
                      << " message=" << message->pDescription << "\n";
        }
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
            message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
        {
            ++summary.errors;
            std::cerr << "D3D12_DEBUG_ERROR id=" << message->ID
                      << " message=" << message->pDescription << "\n";
        }
    }
#else
    (void)device;
#endif
    return summary;
}

std::filesystem::path ExecutableDirectory()
{
    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size())
        throw std::runtime_error("GetModuleFileNameW failed");
    return std::filesystem::path(path.data()).parent_path();
}

ComPtr<ID3DBlob> CompileShader(
    const std::filesystem::path& path,
    const char* entryPoint,
    const char* target)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    const HRESULT result = D3DCompileFromFile(
        path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint, target, flags, 0, &shader, &errors);
    if (FAILED(result))
    {
        std::string message = "shader compilation failed: " + path.string();
        if (errors)
            message += "\n" + std::string(
                static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
        throw std::runtime_error(message);
    }
    return shader;
}

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
    XMFLOAT3 color;
    float emissive = 0.0f;
};

void PushVertex(
    std::vector<Vertex>& vertices,
    const XMFLOAT3& position,
    const XMFLOAT3& normal,
    const XMFLOAT3& color,
    float emissive)
{
    vertices.push_back({position, normal, color, emissive});
}

void PushQuad(
    std::vector<Vertex>& vertices,
    const XMFLOAT3& p0,
    const XMFLOAT3& p1,
    const XMFLOAT3& p2,
    const XMFLOAT3& p3,
    const XMFLOAT3& normal,
    const XMFLOAT3& color,
    float emissive = 0.0f)
{
    PushVertex(vertices, p0, normal, color, emissive);
    PushVertex(vertices, p1, normal, color, emissive);
    PushVertex(vertices, p2, normal, color, emissive);
    PushVertex(vertices, p0, normal, color, emissive);
    PushVertex(vertices, p2, normal, color, emissive);
    PushVertex(vertices, p3, normal, color, emissive);
}

void PushBox(
    std::vector<Vertex>& vertices,
    const XMFLOAT3& minimum,
    const XMFLOAT3& maximum,
    const XMFLOAT3& color,
    float emissive = 0.0f)
{
    const float x0 = minimum.x, y0 = minimum.y, z0 = minimum.z;
    const float x1 = maximum.x, y1 = maximum.y, z1 = maximum.z;
    PushQuad(vertices, {x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1},{0,0,1},color,emissive);
    PushQuad(vertices, {x1,y0,z0},{x0,y0,z0},{x0,y1,z0},{x1,y1,z0},{0,0,-1},color,emissive);
    PushQuad(vertices, {x0,y0,z0},{x0,y0,z1},{x0,y1,z1},{x0,y1,z0},{-1,0,0},color,emissive);
    PushQuad(vertices, {x1,y0,z1},{x1,y0,z0},{x1,y1,z0},{x1,y1,z1},{1,0,0},color,emissive);
    PushQuad(vertices, {x0,y1,z1},{x1,y1,z1},{x1,y1,z0},{x0,y1,z0},{0,1,0},color,emissive);
    PushQuad(vertices, {x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1},{0,-1,0},color,emissive);
}

XMFLOAT3 Subtract(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

XMFLOAT3 Cross(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

XMFLOAT3 Normalize(const XMFLOAT3& value)
{
    const float length = std::max(
        std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z), 1.0e-8f);
    return {value.x/length,value.y/length,value.z/length};
}

void PushTriangle(
    std::vector<Vertex>& vertices,
    const XMFLOAT3& p0,
    const XMFLOAT3& p1,
    const XMFLOAT3& p2,
    const XMFLOAT3& color,
    float emissive = 0.0f)
{
    const XMFLOAT3 normal = Normalize(Cross(Subtract(p1,p0), Subtract(p2,p0)));
    PushVertex(vertices,p0,normal,color,emissive);
    PushVertex(vertices,p1,normal,color,emissive);
    PushVertex(vertices,p2,normal,color,emissive);
}

void PushOctahedron(
    std::vector<Vertex>& vertices,
    const XMFLOAT3& center,
    float radius,
    const XMFLOAT3& color)
{
    const XMFLOAT3 top{center.x,center.y+radius,center.z};
    const XMFLOAT3 bottom{center.x,center.y-radius,center.z};
    const XMFLOAT3 ring[4] = {
        {center.x+radius,center.y,center.z},
        {center.x,center.y,center.z+radius},
        {center.x-radius,center.y,center.z},
        {center.x,center.y,center.z-radius}};
    for (int index = 0; index < 4; ++index)
    {
        const int next = (index + 1) % 4;
        PushTriangle(vertices,top,ring[index],ring[next],color,1.0f);
        PushTriangle(vertices,bottom,ring[next],ring[index],color,1.0f);
    }
}

XMFLOAT3 ToneMap(const XMFLOAT4& value, float exposure)
{
    return {
        std::clamp(1.0f - std::exp(-value.x * exposure), 0.0f, 1.0f),
        std::clamp(1.0f - std::exp(-value.y * exposure), 0.0f, 1.0f),
        std::clamp(1.0f - std::exp(-value.z * exposure), 0.0f, 1.0f)};
}

std::vector<Vertex> BuildSceneVertices(
    const prt::SceneData& scene,
    const std::array<XMFLOAT4, prt::kBrickCount>& brickRadiance,
    const std::array<XMFLOAT4, prt::kReceiverCount>& irradiance,
    const XMFLOAT3& lightPosition)
{
    std::vector<Vertex> vertices;
    vertices.reserve(1200);

    PushQuad(vertices, {-4,-.04f,3.4f},{4,-.04f,3.4f},{4,-.04f,-3.55f},{-4,-.04f,-3.55f},
             {0,1,0},{.12f,.15f,.19f});
    PushQuad(vertices, {-4,0,-3.55f},{4,0,-3.55f},{4,4,-3.55f},{-4,4,-3.55f},
             {0,0,1},{.18f,.15f,.17f});
    PushQuad(vertices, {-4,0,3.4f},{-4,0,-3.55f},{-4,4,-3.55f},{-4,4,3.4f},
             {1,0,0},{.11f,.15f,.19f});
    PushQuad(vertices, {4,0,-3.55f},{4,0,3.4f},{4,4,3.4f},{4,4,-3.55f},
             {-1,0,0},{.11f,.15f,.19f});
    PushBox(vertices, scene.blocker.min, scene.blocker.max, {.2f,.23f,.29f});

    const float receiverStepX = 7.3f / static_cast<float>(prt::kReceiverColumns);
    const float receiverStepZ = 6.05f / static_cast<float>(prt::kReceiverRows);
    for (std::uint32_t index = 0; index < prt::kReceiverCount; ++index)
    {
        const XMFLOAT3 p = scene.receivers[index].position;
        const XMFLOAT3 color = ToneMap(irradiance[index], 4.5f);
        const float gap = .035f;
        PushQuad(vertices,
            {p.x-receiverStepX*.5f+gap,p.y,p.z+receiverStepZ*.5f-gap},
            {p.x+receiverStepX*.5f-gap,p.y,p.z+receiverStepZ*.5f-gap},
            {p.x+receiverStepX*.5f-gap,p.y,p.z-receiverStepZ*.5f+gap},
            {p.x-receiverStepX*.5f+gap,p.y,p.z-receiverStepZ*.5f+gap},
            {0,1,0},color,1.0f);
    }

    for (std::uint32_t index = 0; index < prt::kBrickCount; ++index)
    {
        const prt::Brick& brick = scene.bricks[index];
        const XMFLOAT3 color = ToneMap(brickRadiance[index], 2.2f);
        const float gap = .035f;
        const float x0 = brick.position.x - brick.width*.5f + gap;
        const float x1 = brick.position.x + brick.width*.5f - gap;
        const float y0 = brick.position.y - brick.height*.5f + gap;
        const float y1 = brick.position.y + brick.height*.5f - gap;
        const float z = brick.position.z;
        PushQuad(vertices,{x0,y0,z},{x1,y0,z},{x1,y1,z},{x0,y1,z},{0,0,1},color,1.0f);
    }

    PushBox(vertices,
        {lightPosition.x-.12f,lightPosition.y-.12f,lightPosition.z-.12f},
        {lightPosition.x+.12f,lightPosition.y+.12f,lightPosition.z+.12f},
        {1.0f,.74f,.42f},1.0f);
    const std::uint32_t selected = 1 * prt::kReceiverColumns + prt::kReceiverColumns / 2;
    XMFLOAT3 marker = scene.receivers[selected].position;
    marker.y += .20f;
    PushOctahedron(vertices,marker,.15f,{.15f,1.0f,.92f});
    return vertices;
}

class Dx12PrtLab
{
public:
    int Run(HINSTANCE instance, int showCommand);
    int RunSelfTest();

private:
    static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    void CreateWindowHandle(HINSTANCE instance, int showCommand);
    void CreateCoreDevice();
    void CreateSwapChainAndTargets();
    void CreateComputePipeline();
    void CreateGraphicsPipeline();
    void CreateSceneResources();
    void UpdateBrickRadiance();
    void DispatchAndReadback();
    void RebuildVertexBuffer();
    void Render();
    void Tick();
    void WaitForGpu();
    void UpdateWindowTitle();
    ComPtr<ID3D12Resource> CreateBuffer(
        UINT64 size,
        D3D12_HEAP_TYPE heapType,
        D3D12_RESOURCE_STATES initialState,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);

    HWND m_window = nullptr;
    UINT m_width = kWindowWidth;
    UINT m_height = kWindowHeight;
    bool m_running = true;
    bool m_animate = false;
    bool m_dragging = false;
    POINT m_lastMouse{};
    float m_lightAngle = 35.0f;
    float m_cameraYaw = .58f;
    float m_cameraPitch = .47f;
    float m_cameraZoom = 11.8f;
    std::chrono::steady_clock::time_point m_lastTick = std::chrono::steady_clock::now();

    prt::SceneData m_scene = prt::BuildTeachingScene();
    std::vector<float> m_transfer = prt::BuildTransferMatrix(m_scene);
    std::array<XMFLOAT4, prt::kBrickCount> m_brickRadiance{};
    std::array<XMFLOAT4, prt::kReceiverCount> m_currentIrradiance{};
    std::array<XMFLOAT4, prt::kReceiverCount> m_bakedIrradiance{};

    ComPtr<IDXGIFactory6> m_factory;
    ComPtr<IDXGIAdapter1> m_adapter;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12CommandAllocator> m_allocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;
    bool m_usingWarp = false;
    bool m_debugLayerEnabled = false;

    ComPtr<IDXGISwapChain3> m_swapChain;
    UINT m_frameIndex = 0;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12DescriptorHeap> m_dsvHeap;
    UINT m_rtvStride = 0;
    std::array<ComPtr<ID3D12Resource>, kFrameCount> m_renderTargets;
    ComPtr<ID3D12Resource> m_depth;

    ComPtr<ID3D12RootSignature> m_computeRoot;
    ComPtr<ID3D12PipelineState> m_computePipeline;
    ComPtr<ID3D12DescriptorHeap> m_computeHeap;
    UINT m_computeDescriptorStride = 0;
    ComPtr<ID3D12Resource> m_transferBuffer;
    ComPtr<ID3D12Resource> m_radianceBuffer;
    ComPtr<ID3D12Resource> m_irradianceBuffer;
    ComPtr<ID3D12Resource> m_readbackBuffer;
    std::byte* m_mappedRadiance = nullptr;
    bool m_irradianceReadyForUav = false;

    ComPtr<ID3D12RootSignature> m_graphicsRoot;
    ComPtr<ID3D12PipelineState> m_graphicsPipeline;
    ComPtr<ID3D12Resource> m_vertexBuffer;
    std::byte* m_mappedVertices = nullptr;
    D3D12_VERTEX_BUFFER_VIEW m_vertexView{};
    UINT m_leftVertexCount = 0;
    UINT m_rightVertexCount = 0;
};

ComPtr<ID3D12Resource> Dx12PrtLab::CreateBuffer(
    UINT64 size,
    D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_STATES initialState,
    D3D12_RESOURCE_FLAGS flags)
{
    ComPtr<ID3D12Resource> resource;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(heapType);
    const D3D12_RESOURCE_DESC description = BufferDescription(size, flags);
    DX_CHECK(m_device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, initialState,
        nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}

void Dx12PrtLab::CreateCoreDevice()
{
#if defined(_DEBUG)
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
    {
        debug->EnableDebugLayer();
        m_debugLayerEnabled = true;
    }
#endif
    const UINT factoryFlags = m_debugLayerEnabled ? DXGI_CREATE_FACTORY_DEBUG : 0;
    DX_CHECK(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory)));

    for (UINT index = 0; ; ++index)
    {
        ComPtr<IDXGIAdapter1> candidate;
        if (m_factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 description{};
        candidate->GetDesc1(&description);
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
            continue;
        if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                        __uuidof(ID3D12Device), nullptr)))
        {
            m_adapter = candidate;
            break;
        }
    }
    if (!m_adapter)
    {
        ComPtr<IDXGIAdapter> warp;
        DX_CHECK(m_factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
        DX_CHECK(warp.As(&m_adapter));
        m_usingWarp = true;
    }
    DX_CHECK(D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                               IID_PPV_ARGS(&m_device)));

    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    DX_CHECK(m_device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&m_queue)));
    DX_CHECK(m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator)));
    DX_CHECK(m_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_allocator.Get(), nullptr,
        IID_PPV_ARGS(&m_commandList)));
    DX_CHECK(m_commandList->Close());
    DX_CHECK(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent)
        throw std::runtime_error("CreateEventW failed");
}

void Dx12PrtLab::WaitForGpu()
{
    const UINT64 value = ++m_fenceValue;
    DX_CHECK(m_queue->Signal(m_fence.Get(), value));
    if (m_fence->GetCompletedValue() < value)
    {
        DX_CHECK(m_fence->SetEventOnCompletion(value, m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void Dx12PrtLab::CreateComputePipeline()
{
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {1, &ranges[0]};
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable = {1, &ranges[1]};
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[2].Constants = {0, 0, 2};
    parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC signatureDescription{};
    signatureDescription.NumParameters = 3;
    signatureDescription.pParameters = parameters;
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    DX_CHECK(D3D12SerializeRootSignature(
        &signatureDescription, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &errors));
    DX_CHECK(m_device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&m_computeRoot)));

    const ComPtr<ID3DBlob> computeShader = CompileShader(
        ExecutableDirectory() / L"shaders" / L"PrtCompute.hlsl", "CSMain", "cs_5_1");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDescription{};
    pipelineDescription.pRootSignature = m_computeRoot.Get();
    pipelineDescription.CS = {computeShader->GetBufferPointer(), computeShader->GetBufferSize()};
    DX_CHECK(m_device->CreateComputePipelineState(
        &pipelineDescription, IID_PPV_ARGS(&m_computePipeline)));

    D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
    heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDescription.NumDescriptors = 3;
    heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    DX_CHECK(m_device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&m_computeHeap)));
    m_computeDescriptorStride = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void Dx12PrtLab::CreateSceneResources()
{
    const UINT64 transferBytes = m_transfer.size() * sizeof(float);
    const UINT64 radianceBytes = prt::kBrickCount * sizeof(XMFLOAT4);
    const UINT64 irradianceBytes = prt::kReceiverCount * sizeof(XMFLOAT4);
    m_transferBuffer = CreateBuffer(
        transferBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_radianceBuffer = CreateBuffer(
        radianceBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    m_irradianceBuffer = CreateBuffer(
        irradianceBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    m_readbackBuffer = CreateBuffer(
        irradianceBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    void* mappedTransfer = nullptr;
    const D3D12_RANGE noRead{0,0};
    DX_CHECK(m_transferBuffer->Map(0, &noRead, &mappedTransfer));
    std::memcpy(mappedTransfer, m_transfer.data(), static_cast<size_t>(transferBytes));
    m_transferBuffer->Unmap(0, nullptr);
    void* mappedRadiance = nullptr;
    DX_CHECK(m_radianceBuffer->Map(0, &noRead, &mappedRadiance));
    m_mappedRadiance = static_cast<std::byte*>(mappedRadiance);

    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_computeHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC transferView{};
    transferView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    transferView.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    transferView.Format = DXGI_FORMAT_UNKNOWN;
    transferView.Buffer.NumElements = static_cast<UINT>(m_transfer.size());
    transferView.Buffer.StructureByteStride = sizeof(float);
    m_device->CreateShaderResourceView(m_transferBuffer.Get(), &transferView, handle);

    handle.ptr += m_computeDescriptorStride;
    D3D12_SHADER_RESOURCE_VIEW_DESC radianceView{};
    radianceView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    radianceView.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    radianceView.Format = DXGI_FORMAT_UNKNOWN;
    radianceView.Buffer.NumElements = prt::kBrickCount;
    radianceView.Buffer.StructureByteStride = sizeof(XMFLOAT4);
    m_device->CreateShaderResourceView(m_radianceBuffer.Get(), &radianceView, handle);

    handle.ptr += m_computeDescriptorStride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC irradianceView{};
    irradianceView.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    irradianceView.Format = DXGI_FORMAT_UNKNOWN;
    irradianceView.Buffer.NumElements = prt::kReceiverCount;
    irradianceView.Buffer.StructureByteStride = sizeof(XMFLOAT4);
    m_device->CreateUnorderedAccessView(
        m_irradianceBuffer.Get(), nullptr, &irradianceView, handle);
}

void Dx12PrtLab::UpdateBrickRadiance()
{
    m_brickRadiance = prt::ShadeWallBricks(
        m_scene, prt::LightPositionFromAngle(m_lightAngle), {1.0f,.74f,.42f});
    std::memcpy(m_mappedRadiance, m_brickRadiance.data(), sizeof(m_brickRadiance));
}

void Dx12PrtLab::DispatchAndReadback()
{
    DX_CHECK(m_allocator->Reset());
    DX_CHECK(m_commandList->Reset(m_allocator.Get(), m_computePipeline.Get()));
    if (!m_irradianceReadyForUav)
    {
        auto toUav = TransitionBarrier(
            m_irradianceBuffer.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_commandList->ResourceBarrier(1, &toUav);
        m_irradianceReadyForUav = true;
    }
    ID3D12DescriptorHeap* heaps[] = {m_computeHeap.Get()};
    m_commandList->SetDescriptorHeaps(1, heaps);
    m_commandList->SetComputeRootSignature(m_computeRoot.Get());
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = m_computeHeap->GetGPUDescriptorHandleForHeapStart();
    m_commandList->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += static_cast<UINT64>(2) * m_computeDescriptorStride;
    m_commandList->SetComputeRootDescriptorTable(1, gpu);
    const UINT constants[2] = {prt::kReceiverCount, prt::kBrickCount};
    m_commandList->SetComputeRoot32BitConstants(2, 2, constants, 0);
    m_commandList->Dispatch((prt::kReceiverCount + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = m_irradianceBuffer.Get();
    m_commandList->ResourceBarrier(1, &uav);
    auto toCopy = TransitionBarrier(
        m_irradianceBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_commandList->ResourceBarrier(1, &toCopy);
    m_commandList->CopyResource(m_readbackBuffer.Get(), m_irradianceBuffer.Get());
    auto toUav = TransitionBarrier(
        m_irradianceBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->ResourceBarrier(1, &toUav);
    DX_CHECK(m_commandList->Close());
    ID3D12CommandList* lists[] = {m_commandList.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    WaitForGpu();

    void* data = nullptr;
    const D3D12_RANGE readRange{0, sizeof(m_currentIrradiance)};
    DX_CHECK(m_readbackBuffer->Map(0, &readRange, &data));
    std::memcpy(m_currentIrradiance.data(), data, sizeof(m_currentIrradiance));
    const D3D12_RANGE noWrite{0,0};
    m_readbackBuffer->Unmap(0, &noWrite);
}

void Dx12PrtLab::CreateSwapChainAndTargets()
{
    DXGI_SWAP_CHAIN_DESC1 description{};
    description.Width = m_width;
    description.Height = m_height;
    description.Format = kBackBufferFormat;
    description.SampleDesc = {1,0};
    description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    description.BufferCount = kFrameCount;
    description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> swapChain1;
    DX_CHECK(m_factory->CreateSwapChainForHwnd(
        m_queue.Get(), m_window, &description, nullptr, nullptr, &swapChain1));
    DX_CHECK(m_factory->MakeWindowAssociation(m_window, DXGI_MWA_NO_ALT_ENTER));
    DX_CHECK(swapChain1.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDescription{};
    rtvHeapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDescription.NumDescriptors = kFrameCount;
    DX_CHECK(m_device->CreateDescriptorHeap(&rtvHeapDescription, IID_PPV_ARGS(&m_rtvHeap)));
    m_rtvStride = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT index = 0; index < kFrameCount; ++index)
    {
        DX_CHECK(m_swapChain->GetBuffer(index, IID_PPV_ARGS(&m_renderTargets[index])));
        m_device->CreateRenderTargetView(m_renderTargets[index].Get(), nullptr, rtv);
        rtv.ptr += m_rtvStride;
    }

    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDescription{};
    dsvHeapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDescription.NumDescriptors = 1;
    DX_CHECK(m_device->CreateDescriptorHeap(&dsvHeapDescription, IID_PPV_ARGS(&m_dsvHeap)));
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D32_FLOAT;
    clear.DepthStencil.Depth = 1.0f;
    D3D12_RESOURCE_DESC depthDescription{};
    depthDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDescription.Width = m_width;
    depthDescription.Height = m_height;
    depthDescription.DepthOrArraySize = 1;
    depthDescription.MipLevels = 1;
    depthDescription.Format = DXGI_FORMAT_D32_FLOAT;
    depthDescription.SampleDesc = {1,0};
    depthDescription.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depthDescription.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    DX_CHECK(m_device->CreateCommittedResource(
        &defaultHeap,D3D12_HEAP_FLAG_NONE,&depthDescription,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(&m_depth)));
    m_device->CreateDepthStencilView(
        m_depth.Get(), nullptr, m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

void Dx12PrtLab::CreateGraphicsPipeline()
{
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants = {0,0,16};
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC signatureDescription{};
    signatureDescription.NumParameters = 1;
    signatureDescription.pParameters = &parameter;
    signatureDescription.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    DX_CHECK(D3D12SerializeRootSignature(
        &signatureDescription,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&errors));
    DX_CHECK(m_device->CreateRootSignature(
        0,serialized->GetBufferPointer(),serialized->GetBufferSize(),
        IID_PPV_ARGS(&m_graphicsRoot)));

    const auto shaderPath = ExecutableDirectory() / L"shaders" / L"Scene.hlsl";
    const ComPtr<ID3DBlob> vertexShader = CompileShader(shaderPath,"VSMain","vs_5_1");
    const ComPtr<ID3DBlob> pixelShader = CompileShader(shaderPath,"PSMain","ps_5_1");
    const D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,normal),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32_FLOAT,0,offsetof(Vertex,emissive),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = m_graphicsRoot.Get();
    pipeline.VS = {vertexShader->GetBufferPointer(),vertexShader->GetBufferSize()};
    pipeline.PS = {pixelShader->GetBufferPointer(),pixelShader->GetBufferSize()};
    pipeline.InputLayout = {inputLayout,_countof(inputLayout)};
    pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.SampleMask = UINT_MAX;
    pipeline.NumRenderTargets = 1;
    pipeline.RTVFormats[0] = kBackBufferFormat;
    pipeline.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    pipeline.SampleDesc = {1,0};
    pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipeline.RasterizerState.DepthClipEnable = TRUE;
    pipeline.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pipeline.DepthStencilState.DepthEnable = TRUE;
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    pipeline.DepthStencilState.StencilEnable = FALSE;
    DX_CHECK(m_device->CreateGraphicsPipelineState(&pipeline,IID_PPV_ARGS(&m_graphicsPipeline)));

    constexpr UINT64 vertexCapacity = 512 * 1024;
    m_vertexBuffer = CreateBuffer(
        vertexCapacity,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE noRead{0,0};
    DX_CHECK(m_vertexBuffer->Map(0,&noRead,&mapped));
    m_mappedVertices = static_cast<std::byte*>(mapped);
    m_vertexView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
    m_vertexView.SizeInBytes = static_cast<UINT>(vertexCapacity);
    m_vertexView.StrideInBytes = sizeof(Vertex);
}

void Dx12PrtLab::RebuildVertexBuffer()
{
    const XMFLOAT3 light = prt::LightPositionFromAngle(m_lightAngle);
    std::vector<Vertex> left = BuildSceneVertices(
        m_scene,m_brickRadiance,m_bakedIrradiance,light);
    std::vector<Vertex> right = BuildSceneVertices(
        m_scene,m_brickRadiance,m_currentIrradiance,light);
    m_leftVertexCount = static_cast<UINT>(left.size());
    m_rightVertexCount = static_cast<UINT>(right.size());
    const size_t totalBytes = (left.size()+right.size())*sizeof(Vertex);
    if (totalBytes > m_vertexView.SizeInBytes)
        throw std::runtime_error("teaching vertex buffer capacity exceeded");
    std::memcpy(m_mappedVertices,left.data(),left.size()*sizeof(Vertex));
    std::memcpy(m_mappedVertices+left.size()*sizeof(Vertex),right.data(),right.size()*sizeof(Vertex));
}

void Dx12PrtLab::Render()
{
    UpdateBrickRadiance();
    DispatchAndReadback();
    RebuildVertexBuffer();

    DX_CHECK(m_allocator->Reset());
    DX_CHECK(m_commandList->Reset(m_allocator.Get(),m_graphicsPipeline.Get()));
    auto toRenderTarget = TransitionBarrier(
        m_renderTargets[m_frameIndex].Get(),D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_commandList->ResourceBarrier(1,&toRenderTarget);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += static_cast<UINT64>(m_frameIndex)*m_rtvStride;
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    const float clearColor[4] = {.018f,.027f,.044f,1.0f};
    m_commandList->OMSetRenderTargets(1,&rtv,FALSE,&dsv);
    m_commandList->ClearRenderTargetView(rtv,clearColor,0,nullptr);
    m_commandList->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,1.0f,0,0,nullptr);
    m_commandList->SetGraphicsRootSignature(m_graphicsRoot.Get());
    m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_commandList->IASetVertexBuffers(0,1,&m_vertexView);

    const XMVECTOR eye = XMVectorSet(
        m_cameraZoom*std::cos(m_cameraPitch)*std::sin(m_cameraYaw),
        m_cameraZoom*std::sin(m_cameraPitch)+1.1f,
        m_cameraZoom*std::cos(m_cameraPitch)*std::cos(m_cameraYaw),1.0f);
    const XMMATRIX view = XMMatrixLookAtLH(eye,XMVectorSet(0,1.35f,-.2f,1),XMVectorSet(0,1,0,0));
    const XMMATRIX projection = XMMatrixPerspectiveFovLH(.79f,
        (static_cast<float>(m_width)*.5f)/static_cast<float>(m_height),.1f,60.0f);
    XMFLOAT4X4 viewProjection{};
    // Scene.hlsl uses column-vector multiplication: mul(M, position).
    // DirectXMath stores a row-vector matrix in row-major memory; HLSL's
    // default column-major constant layout therefore performs the required
    // transpose while reading these 16 root constants.  Transposing here as
    // well would transpose twice and produce the giant/clipped triangles that
    // make the teaching scene unreadable.
    XMStoreFloat4x4(&viewProjection,view*projection);
    m_commandList->SetGraphicsRoot32BitConstants(0,16,&viewProjection,0);

    D3D12_VIEWPORT viewport{0,0,static_cast<float>(m_width)*.5f,static_cast<float>(m_height),0,1};
    D3D12_RECT scissor{0,0,static_cast<LONG>(m_width/2),static_cast<LONG>(m_height)};
    m_commandList->RSSetViewports(1,&viewport);
    m_commandList->RSSetScissorRects(1,&scissor);
    m_commandList->DrawInstanced(m_leftVertexCount,1,0,0);
    viewport.TopLeftX = static_cast<float>(m_width)*.5f;
    scissor.left = static_cast<LONG>(m_width/2);
    scissor.right = static_cast<LONG>(m_width);
    m_commandList->RSSetViewports(1,&viewport);
    m_commandList->RSSetScissorRects(1,&scissor);
    m_commandList->DrawInstanced(m_rightVertexCount,1,m_leftVertexCount,0);

    auto toPresent = TransitionBarrier(
        m_renderTargets[m_frameIndex].Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PRESENT);
    m_commandList->ResourceBarrier(1,&toPresent);
    DX_CHECK(m_commandList->Close());
    ID3D12CommandList* lists[] = {m_commandList.Get()};
    m_queue->ExecuteCommandLists(1,lists);
    DX_CHECK(m_swapChain->Present(1,0));
    WaitForGpu();
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}

void Dx12PrtLab::Tick()
{
    const auto now = std::chrono::steady_clock::now();
    const float delta = std::chrono::duration<float>(now-m_lastTick).count();
    m_lastTick = now;
    if (m_animate)
    {
        m_lightAngle += delta*34.0f;
        if (m_lightAngle > 150.0f)
            m_lightAngle = -150.0f;
    }
    Render();
    UpdateWindowTitle();
}

void Dx12PrtLab::UpdateWindowTitle()
{
    const std::uint32_t selected = 1*prt::kReceiverColumns+prt::kReceiverColumns/2;
    std::wostringstream title;
    title << L"DX12 PRT Probe Lab | 左: baked answer  右: GPU W·R | angle="
          << static_cast<int>(m_lightAngle) << L"° | old="
          << prt::Luminance(m_bakedIrradiance[selected]) << L" current="
          << prt::Luminance(m_currentIrradiance[selected])
          << (m_usingWarp ? L" | WARP" : L" | hardware");
    SetWindowTextW(m_window,title.str().c_str());
}

LRESULT CALLBACK Dx12PrtLab::WindowProc(
    HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    Dx12PrtLab* app = reinterpret_cast<Dx12PrtLab*>(GetWindowLongPtrW(window,GWLP_USERDATA));
    if (message == WM_NCCREATE)
    {
        auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
        app = static_cast<Dx12PrtLab*>(create->lpCreateParams);
        SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(app));
    }
    return app ? app->HandleMessage(window,message,wParam,lParam)
               : DefWindowProcW(window,message,wParam,lParam);
}

LRESULT Dx12PrtLab::HandleMessage(
    HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) DestroyWindow(window);
        else if (wParam == 'A' || wParam == VK_LEFT) m_lightAngle = std::max(-150.0f,m_lightAngle-3.0f);
        else if (wParam == 'D' || wParam == VK_RIGHT) m_lightAngle = std::min(150.0f,m_lightAngle+3.0f);
        else if (wParam == VK_SPACE) m_animate = !m_animate;
        else if (wParam == 'R') m_bakedIrradiance = m_currentIrradiance;
        return 0;
    case WM_LBUTTONDOWN:
        m_dragging = true;
        m_lastMouse = {GET_X_LPARAM(lParam),GET_Y_LPARAM(lParam)};
        SetCapture(window);
        return 0;
    case WM_MOUSEMOVE:
        if (m_dragging)
        {
            const POINT current{GET_X_LPARAM(lParam),GET_Y_LPARAM(lParam)};
            m_cameraYaw += static_cast<float>(current.x-m_lastMouse.x)*.008f;
            m_cameraPitch = std::clamp(
                m_cameraPitch+static_cast<float>(current.y-m_lastMouse.y)*.006f,.14f,1.18f);
            m_lastMouse = current;
        }
        return 0;
    case WM_LBUTTONUP:
        m_dragging = false;
        ReleaseCapture();
        return 0;
    case WM_MOUSEWHEEL:
        m_cameraZoom = std::clamp(
            m_cameraZoom*std::exp(-static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam))/1200.0f),
            7.2f,18.0f);
        return 0;
    case WM_CLOSE:
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        m_running = false;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(window,message,wParam,lParam);
    }
}

void Dx12PrtLab::CreateWindowHandle(HINSTANCE instance, int showCommand)
{
    const wchar_t* className = L"DX12_PRT_PROBE_LAB";
    WNDCLASSEXW windowClass{sizeof(WNDCLASSEXW)};
    windowClass.style = CS_HREDRAW|CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr,IDC_ARROW);
    windowClass.lpszClassName = className;
    if (!RegisterClassExW(&windowClass))
        throw std::runtime_error("RegisterClassExW failed");
    RECT rectangle{0,0,static_cast<LONG>(m_width),static_cast<LONG>(m_height)};
    const DWORD style = WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX;
    AdjustWindowRect(&rectangle,style,FALSE);
    m_window = CreateWindowExW(0,className,L"DX12 PRT Probe Lab",style,
        CW_USEDEFAULT,CW_USEDEFAULT,rectangle.right-rectangle.left,rectangle.bottom-rectangle.top,
        nullptr,nullptr,instance,this);
    if (!m_window)
        throw std::runtime_error("CreateWindowExW failed");
    ShowWindow(m_window,showCommand);
}

int Dx12PrtLab::Run(HINSTANCE instance, int showCommand)
{
    CreateWindowHandle(instance,showCommand);
    CreateCoreDevice();
    CreateComputePipeline();
    CreateSceneResources();
    CreateSwapChainAndTargets();
    CreateGraphicsPipeline();
    UpdateBrickRadiance();
    DispatchAndReadback();
    m_bakedIrradiance = m_currentIrradiance;
    MSG message{};
    while (m_running)
    {
        while (PeekMessageW(&message,nullptr,0,0,PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (m_running)
            Tick();
    }
    WaitForGpu();
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    return 0;
}

int Dx12PrtLab::RunSelfTest()
{
    CreateCoreDevice();
    CreateComputePipeline();
    CreateSceneResources();
    UpdateBrickRadiance();
    DispatchAndReadback();
    const auto reference = prt::EvaluateTransferCpu(m_transfer,m_brickRadiance);
    float maximumError = 0.0f;
    for (std::uint32_t index = 0; index < prt::kReceiverCount; ++index)
    {
        maximumError = std::max(maximumError,std::abs(reference[index].x-m_currentIrradiance[index].x));
        maximumError = std::max(maximumError,std::abs(reference[index].y-m_currentIrradiance[index].y));
        maximumError = std::max(maximumError,std::abs(reference[index].z-m_currentIrradiance[index].z));
    }
    DXGI_ADAPTER_DESC1 description{};
    m_adapter->GetDesc1(&description);
    std::wcout << L"adapter=" << description.Description
               << L" backend=" << (m_usingWarp?L"WARP":L"hardware") << L"\n";
    std::cout << "receivers=" << prt::kReceiverCount
              << " bricks=" << prt::kBrickCount
              << " max_abs_error=" << maximumError << "\n";
    const DebugMessageSummary debugMessages = InspectDebugMessages(m_device.Get());
    std::cout << "d3d12_debug_layer=" << (m_debugLayerEnabled ? "enabled" : "disabled")
              << " stored_messages=" << debugMessages.stored
              << " warnings=" << debugMessages.warnings
              << " errors=" << debugMessages.errors << "\n";
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    if (debugMessages.errors != 0)
    {
        std::cerr << "SELF_TEST_FAILED_D3D12_DEBUG\n";
        return 3;
    }
    if (maximumError > 1.0e-5f)
    {
        std::cerr << "SELF_TEST_FAILED\n";
        return 2;
    }
    std::cout << "SELF_TEST_PASSED\n";
    return 0;
}
}

int wmain(int argc, wchar_t** argv)
{
    try
    {
        const bool selfTest = argc > 1 && std::wstring_view(argv[1]) == L"--self-test";
        // Keep stdout/stderr for the CLI self-test, but normal interactive runs
        // should present only the teaching window (including when double-clicked).
        if (!selfTest)
            FreeConsole();
        Dx12PrtLab app;
        return selfTest ? app.RunSelfTest()
                        : app.Run(GetModuleHandleW(nullptr),SW_SHOWDEFAULT);
    }
    catch (const std::exception& error)
    {
        std::cerr << "DX12_GI_Lab fatal error: " << error.what() << "\n";
        MessageBoxA(nullptr,error.what(),"DX12_GI_Lab fatal error",MB_OK|MB_ICONERROR);
        return 1;
    }
}
