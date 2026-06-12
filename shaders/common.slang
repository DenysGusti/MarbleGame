#ifndef COMMON_SLANG
#define COMMON_SLANG

struct CameraUBO {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 lightDir;
    float4 baseColor;
};

#endif // COMMON_SLANG
