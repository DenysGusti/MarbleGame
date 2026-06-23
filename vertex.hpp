#ifndef MARBLEGAME_VERTEX_HPP
#define MARBLEGAME_VERTEX_HPP

#include <array>
#include <cstddef>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    bool operator==(const Vertex &other) const {
        return position == other.position &&
               normal == other.normal &&
               texCoord == other.texCoord &&
               tangent == other.tangent;
    }

    [[nodiscard]] static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription(
            0, // binding
            sizeof(Vertex), // stride
            vk::VertexInputRate::eVertex // inputRate
        );
        return bindingDescription;
    }

    [[nodiscard]] static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        constexpr std::array attributeDescriptions = {
            vk::VertexInputAttributeDescription{
                .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, position)
            },
            vk::VertexInputAttributeDescription{
                .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)
            },
            vk::VertexInputAttributeDescription{
                .location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)
            },
            vk::VertexInputAttributeDescription{
                .location = 3, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = offsetof(Vertex, tangent)
            }
        };
        return attributeDescriptions;
    }
};

#endif // MARBLEGAME_VERTEX_HPP
