#ifndef COMMON_SLANG
#define COMMON_SLANG

enum TextureIndex : int {
  Ball = 0,
  Floor = 1,
  Goal = 2,
  None = -1
};

struct CameraUBO {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
  float4 lightDir;
  float4 baseColor;
};

struct PushConstants {
  float4x4 model;           // Takes 64 bytes
  float3 size;             // Takes 12 bytes
  TextureIndex textureIndex; // Takes 4 bytes
};

#endif // COMMON_SLANG
