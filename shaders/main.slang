#include "common.slang"

// Bind to descriptor set 0, binding 0
[[vk::binding(0, 0)]]
ConstantBuffer<CameraUBO> ubo;

// 2. Vertex Input matching the C++ Vertex struct
struct VertexInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

// 3. Data passed from Vertex Shader to Fragment Shader
struct VertexOutput {
    float4 position : SV_Position;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD;
};

// ==========================================
// --- VERTEX SHADER ---
// ==========================================
[shader("vertex")]
VertexOutput vertexMain(VertexInput input) {
    VertexOutput output;

    // Apply standard projection to grayscale vertices (i.e. the sphere)
    if (input.color.r == input.color.g && input.color.g == input.color.b) {
        float4 worldPos = mul(ubo.model, float4(input.position, 1.0));
        output.position = mul(ubo.proj, mul(ubo.view, worldPos));
        output.normal = mul((float3x3)ubo.model, input.normal);
    } else {
        // Debug lines projection: rotate to view space and offset in front of camera
        float3 viewPos = mul((float3x3)ubo.view, input.position) + float3(0.0, 0.0, -0.5);
        output.position = mul(ubo.proj, float4(viewPos, 1.0));
        output.normal = input.normal;
    }
    
    // Pass vertex color and texture coordinates
    output.color = input.color;
    output.uv = input.uv;

    return output;
}

// ==========================================
// --- FRAGMENT SHADER ---
// ==========================================
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 baseColor = input.color;
    float diff = 1.0;

    // Apply lighting and UBO base color only to grayscale vertices (i.e. the sphere)
    if (input.color.r == input.color.g && input.color.g == input.color.b) {
        baseColor = input.color * ubo.baseColor.rgb;
        
        // Read directional light direction from uniform buffer
        float3 lightDir = normalize(ubo.lightDir.xyz);
        float3 norm = normalize(input.normal);

        // Calculate diffuse lighting with an ambient base of 0.2
        diff = max(dot(norm, lightDir), 0.2);
    }

    // Apply lighting to color
    float3 finalColor = baseColor * diff;

    return float4(finalColor, 1.0);
}