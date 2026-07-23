cbuffer Camera : register(b0)
{
    float4x4 gViewProjection;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR0;
    float emissive : TEXCOORD0;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 color : COLOR0;
    float emissive : TEXCOORD0;
};

PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = mul(gViewProjection, float4(input.position, 1.0));
    output.normal = input.normal;
    output.color = input.color;
    output.emissive = input.emissive;
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    const float3 keyDirection = normalize(float3(-0.3, 0.85, 0.45));
    const float ndotl = max(dot(normalize(input.normal), keyDirection), 0.0);
    const float shade = lerp(0.32 + 0.68 * ndotl, 1.0, saturate(input.emissive));
    const float3 linearColor = max(input.color * shade, 0.0);
    return float4(pow(linearColor, 1.0 / 2.2), 1.0);
}

