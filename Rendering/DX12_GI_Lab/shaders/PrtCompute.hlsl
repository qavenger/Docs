// Same equation as the HTML/WGSL demo: E[p] = sum_b W[p,b] * R[b].
StructuredBuffer<float>  gTransfer : register(t0);
StructuredBuffer<float4> gBrickRadiance : register(t1);
RWStructuredBuffer<float4> gReceiverIrradiance : register(u0);

cbuffer Params : register(b0)
{
    uint gReceiverCount;
    uint gBrickCount;
};

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint receiver = dispatchThreadId.x;
    if (receiver >= gReceiverCount)
        return;

    float3 sum = 0.0;
    const uint row = receiver * gBrickCount;
    for (uint brick = 0; brick < gBrickCount; ++brick)
        sum += gTransfer[row + brick] * gBrickRadiance[brick].rgb;

    gReceiverIrradiance[receiver] = float4(sum, 1.0);
}

