#ifndef COMMON_HPP
#define COMMON_HPP

#include <glm/glm.hpp>

struct CameraUBO {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 lightDir;
    glm::vec4 baseColor;
};

#endif // COMMON_HPP