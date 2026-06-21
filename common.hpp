#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <glm/glm.hpp>

enum class TextureIndex : std::int32_t {
    Ball = 0,
    Floor = 1,
    None = -1
};

struct CameraUBO {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 lightDir;
    glm::vec4 baseColor;
};

struct PushConstants {
    glm::mat4 model;
    TextureIndex textureIndex;
    std::int32_t pad0;
    std::int32_t pad1;
    std::int32_t pad2;
};

#endif // COMMON_HPP