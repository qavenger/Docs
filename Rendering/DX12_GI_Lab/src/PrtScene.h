#pragma once

#include <DirectXMath.h>
#include <array>
#include <cstdint>
#include <vector>

namespace prt
{
constexpr std::uint32_t kBrickColumns = 8;
constexpr std::uint32_t kBrickRows = 5;
constexpr std::uint32_t kBrickCount = kBrickColumns * kBrickRows;
constexpr std::uint32_t kReceiverColumns = 12;
constexpr std::uint32_t kReceiverRows = 10;
constexpr std::uint32_t kReceiverCount = kReceiverColumns * kReceiverRows;

struct Aabb
{
    DirectX::XMFLOAT3 min;
    DirectX::XMFLOAT3 max;
};

struct Brick
{
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
    DirectX::XMFLOAT3 albedo;
    float area = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

struct Receiver
{
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
};

struct SceneData
{
    std::array<Brick, kBrickCount> bricks;
    std::array<Receiver, kReceiverCount> receivers;
    Aabb blocker;
};

SceneData BuildTeachingScene();
bool SegmentIntersectsAabb(
    const DirectX::XMFLOAT3& a,
    const DirectX::XMFLOAT3& b,
    const Aabb& aabb);
std::vector<float> BuildTransferMatrix(const SceneData& scene);
std::array<DirectX::XMFLOAT4, kBrickCount> ShadeWallBricks(
    const SceneData& scene,
    const DirectX::XMFLOAT3& lightPosition,
    const DirectX::XMFLOAT3& lightRgb);
std::array<DirectX::XMFLOAT4, kReceiverCount> EvaluateTransferCpu(
    const std::vector<float>& transfer,
    const std::array<DirectX::XMFLOAT4, kBrickCount>& brickRadiance);
DirectX::XMFLOAT3 LightPositionFromAngle(float degrees);
float Luminance(const DirectX::XMFLOAT4& color);
}

