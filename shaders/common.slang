#ifndef COMMON_SLANG
#define COMMON_SLANG

enum TextureIndex : int {
  Ball = 0,
  Floor = 1,
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
  float4x4 model;
  TextureIndex textureIndex;
  int pad0;
  int pad1;
  int pad2;
};

#endif // COMMON_SLANG
