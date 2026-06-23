#include "common.slang"

// Bind to descriptor set 0, binding 0
[[vk::binding(0, 0)]]
ConstantBuffer<CameraUBO> ubo;

// Declare push constants
[[vk::push_constant]]
PushConstants pcs;

// Texture bindings (combined image samplers)
[[vk::binding(1, 0)]]
Sampler2D ballTex;
[[vk::binding(2, 0)]]
Sampler2D floorTex;

// 2. Vertex Input matching the C++ Vertex struct
struct VertexInput {
  float3 position : POSITION;
  float3 normal : NORMAL;
  float2 uv : TEXCOORD;
  float4 tangent : TANGENT;
};

// 3. Data passed from Vertex Shader to Fragment Shader
struct VertexOutput {
  float4 position : SV_Position;
  float3 normal : NORMAL;
  float3 viewPos : VIEW_POS;
  float3 color : COLOR;
  float2 uv : TEXCOORD;
};

// ==========================================
// --- VERTEX SHADER ---
// ==========================================
[shader("vertex")]
VertexOutput vertexMain(VertexInput input) {
  VertexOutput output;

  // Use textureIndex to determine if it is a debug line or geometry
  if (pcs.textureIndex != TextureIndex::None) {
    float4 worldPos = mul(pcs.model, float4(input.position, 1.0));
    float4 viewPos4 = mul(ubo.view, worldPos);
    output.position = mul(ubo.proj, viewPos4);
    output.viewPos = viewPos4.xyz;
    output.normal =
        mul((float3x3)ubo.view, mul((float3x3)pcs.model, input.normal));
  } else {
    // Debug lines projection: rotate to view space and offset in front of
    // camera
    float3 viewPos =
        mul((float3x3)ubo.view, input.position) + float3(0.0, 0.0, -0.5);
    output.position = mul(ubo.proj, float4(viewPos, 1.0));
    output.viewPos = viewPos;
    output.normal = input.normal;
  }

  // Pass vertex color (mapped from tangent for debug lines) and texture coordinates
  if (pcs.textureIndex != TextureIndex::None) {
    output.color = float3(1.0, 1.0, 1.0);
  } else {
    output.color = input.tangent.xyz;
  }
  output.uv = input.uv;

  return output;
}

// ==========================================
// --- FRAGMENT SHADER ---
// ==========================================
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
  float3 baseColor = input.color;

  if (pcs.textureIndex == TextureIndex::Ball) {
    baseColor = ballTex.Sample(input.uv).rgb;
  } else if (pcs.textureIndex == TextureIndex::Floor) {
    baseColor = floorTex.Sample(input.uv).rgb;
  }

  float3 finalColor = baseColor;

  // Apply lighting and specular highlights only to textured geometry
  if (pcs.textureIndex != TextureIndex::None) {
    // Read directional light direction from uniform buffer and transform to
    // view space
    float3 lightDirView = normalize(mul((float3x3)ubo.view, ubo.lightDir.xyz));
    float3 norm = normalize(input.normal);

    // Calculate diffuse lighting
    float diff = max(dot(norm, lightDirView), 0.0);
    float3 diffuseColor = diff * baseColor;

    // Calculate ambient lighting (ambient base of 0.2)
    float3 ambientColor = 0.2 * baseColor;

    // Calculate specular lighting (Blinn-Phong)
    float3 viewDir = normalize(-input.viewPos);
    float3 halfDir = normalize(lightDirView + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
    float3 specularColor =
        0.5 * spec *
        float3(1.0, 1.0, 1.0); // specular strength = 0.5, white highlight

    finalColor = ambientColor + diffuseColor + specularColor;
  }

  return float4(finalColor, 1.0);
}
