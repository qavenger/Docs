#include "PrtScene.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

using namespace DirectX;

namespace
{
XMFLOAT3 Subtract(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float Dot(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float Length(const XMFLOAT3& value)
{
    return std::sqrt(Dot(value, value));
}

XMFLOAT3 Normalize(const XMFLOAT3& value)
{
    const float length = std::max(Length(value), 1.0e-8f);
    return {value.x / length, value.y / length, value.z / length};
}
}

namespace prt
{
SceneData BuildTeachingScene()
{
    SceneData scene{};
    scene.blocker = {{-0.65f, 0.0f, -0.55f}, {0.65f, 1.65f, 0.65f}};

    const float receiverStepX = 7.3f / static_cast<float>(kReceiverColumns);
    const float receiverStepZ = 6.05f / static_cast<float>(kReceiverRows);
    for (std::uint32_t z = 0; z < kReceiverRows; ++z)
    {
        for (std::uint32_t x = 0; x < kReceiverColumns; ++x)
        {
            const std::uint32_t index = z * kReceiverColumns + x;
            scene.receivers[index] = {
                {-3.65f + (static_cast<float>(x) + 0.5f) * receiverStepX,
                 0.025f,
                 -3.05f + (static_cast<float>(z) + 0.5f) * receiverStepZ},
                {0.0f, 1.0f, 0.0f}};
        }
    }

    const float brickWidth = 7.1f / static_cast<float>(kBrickColumns);
    const float brickHeight = 3.25f / static_cast<float>(kBrickRows);
    for (std::uint32_t y = 0; y < kBrickRows; ++y)
    {
        for (std::uint32_t x = 0; x < kBrickColumns; ++x)
        {
            const std::uint32_t index = y * kBrickColumns + x;
            const bool blue = (x == 1 && y > 1) || (x == 6 && y < 3);
            scene.bricks[index] = {
                {-3.55f + (static_cast<float>(x) + 0.5f) * brickWidth,
                 0.28f + (static_cast<float>(y) + 0.5f) * brickHeight,
                 -3.47f},
                {0.0f, 0.0f, 1.0f},
                blue ? XMFLOAT3{0.18f, 0.42f, 0.75f}
                     : XMFLOAT3{0.78f, 0.25f + 0.04f * static_cast<float>(y),
                                0.16f + 0.03f * static_cast<float>(x)},
                brickWidth * brickHeight * 0.86f,
                brickWidth,
                brickHeight};
        }
    }
    return scene;
}

bool SegmentIntersectsAabb(const XMFLOAT3& a, const XMFLOAT3& b, const Aabb& aabb)
{
    const XMFLOAT3 direction = Subtract(b, a);
    float tMin = 0.0f;
    float tMax = 1.0f;
    const float origin[3] = {a.x, a.y, a.z};
    const float delta[3] = {direction.x, direction.y, direction.z};
    const float minimum[3] = {aabb.min.x, aabb.min.y, aabb.min.z};
    const float maximum[3] = {aabb.max.x, aabb.max.y, aabb.max.z};

    for (int axis = 0; axis < 3; ++axis)
    {
        if (std::abs(delta[axis]) < 1.0e-8f)
        {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis])
                return false;
            continue;
        }
        float t0 = (minimum[axis] - origin[axis]) / delta[axis];
        float t1 = (maximum[axis] - origin[axis]) / delta[axis];
        if (t0 > t1)
            std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMin > tMax)
            return false;
    }
    return tMax > 0.001f && tMin < 0.999f;
}

std::vector<float> BuildTransferMatrix(const SceneData& scene)
{
    std::vector<float> transfer(kReceiverCount * kBrickCount, 0.0f);
    for (std::uint32_t receiverIndex = 0; receiverIndex < kReceiverCount; ++receiverIndex)
    {
        const Receiver& receiver = scene.receivers[receiverIndex];
        for (std::uint32_t brickIndex = 0; brickIndex < kBrickCount; ++brickIndex)
        {
            const Brick& brick = scene.bricks[brickIndex];
            const XMFLOAT3 toBrick = Subtract(brick.position, receiver.position);
            const float distance = Length(toBrick);
            const XMFLOAT3 direction = Normalize(toBrick);
            const XMFLOAT3 reverseDirection{-direction.x, -direction.y, -direction.z};
            const float cosineReceiver = std::max(0.0f, Dot(receiver.normal, direction));
            const float cosineBrick = std::max(0.0f, Dot(brick.normal, reverseDirection));
            const bool blocked = SegmentIntersectsAabb(receiver.position, brick.position, scene.blocker);
            if (!blocked)
            {
                // brickRadiance already contains the Lambert albedo / pi term.
                // Radiance-to-irradiance geometry therefore has no second / pi.
                transfer[receiverIndex * kBrickCount + brickIndex] =
                    cosineReceiver * cosineBrick * brick.area /
                    (distance * distance);
            }
        }
    }
    return transfer;
}

std::array<XMFLOAT4, kBrickCount> ShadeWallBricks(
    const SceneData& scene,
    const XMFLOAT3& lightPosition,
    const XMFLOAT3& lightRgb)
{
    std::array<XMFLOAT4, kBrickCount> radiance{};
    for (std::uint32_t index = 0; index < kBrickCount; ++index)
    {
        const Brick& brick = scene.bricks[index];
        const XMFLOAT3 toLight = Subtract(lightPosition, brick.position);
        const float distanceSquared = std::max(0.1f, Dot(toLight, toLight));
        const float ndotl = std::max(0.0f, Dot(brick.normal, Normalize(toLight)));
        const float power = 5.6f * ndotl / (1.0f + 0.12f * distanceSquared);
        radiance[index] = {
            brick.albedo.x * lightRgb.x * power / std::numbers::pi_v<float> + 0.006f * brick.albedo.x,
            brick.albedo.y * lightRgb.y * power / std::numbers::pi_v<float> + 0.006f * brick.albedo.y,
            brick.albedo.z * lightRgb.z * power / std::numbers::pi_v<float> + 0.006f * brick.albedo.z,
            1.0f};
    }
    return radiance;
}

std::array<XMFLOAT4, kReceiverCount> EvaluateTransferCpu(
    const std::vector<float>& transfer,
    const std::array<XMFLOAT4, kBrickCount>& brickRadiance)
{
    if (transfer.size() != kReceiverCount * kBrickCount)
        throw std::invalid_argument("transfer matrix has an unexpected size");

    std::array<XMFLOAT4, kReceiverCount> irradiance{};
    for (std::uint32_t receiver = 0; receiver < kReceiverCount; ++receiver)
    {
        XMFLOAT4 sum{0.0f, 0.0f, 0.0f, 1.0f};
        for (std::uint32_t brick = 0; brick < kBrickCount; ++brick)
        {
            const float weight = transfer[receiver * kBrickCount + brick];
            sum.x += weight * brickRadiance[brick].x;
            sum.y += weight * brickRadiance[brick].y;
            sum.z += weight * brickRadiance[brick].z;
        }
        irradiance[receiver] = sum;
    }
    return irradiance;
}

XMFLOAT3 LightPositionFromAngle(float degrees)
{
    const float radians = degrees * std::numbers::pi_v<float> / 180.0f;
    // Keep the teaching light close to the colored back-wall bricks.  The
    // smaller z arc makes their changing radiance—and therefore E = W * R—
    // immediately visible in the floor receivers on both halves of the demo.
    return {3.15f * std::sin(radians), 2.75f, -0.90f + 0.45f * std::cos(radians)};
}

float Luminance(const XMFLOAT4& color)
{
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}
}
