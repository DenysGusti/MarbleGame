#ifndef MARBLEGAME_INSTANCE_DATA_HPP
#define MARBLEGAME_INSTANCE_DATA_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

struct InstanceData {
    glm::mat4 modelMatrix{};
    glm::mat3x4 normalMatrix{};
    std::uint32_t materialIndex{0};

    InstanceData() {
        modelMatrix = glm::mat4{1.0f};
        normalMatrix[0] = glm::vec4{1.0f, 0.0f, 0.0f, 0.0f};
        normalMatrix[1] = glm::vec4{0.0f, 1.0f, 0.0f, 0.0f};
        normalMatrix[2] = glm::vec4{0.0f, 0.0f, 1.0f, 0.0f};
        materialIndex = 0;
    }

    explicit InstanceData(const glm::mat4 &transform, const std::uint32_t matIndex = 0) {
        modelMatrix = transform;

        const glm::mat3 normalMat3 = glm::transpose(glm::inverse(glm::mat3{transform}));
        normalMatrix[0] = glm::vec4{normalMat3[0], 0.0f};
        normalMatrix[1] = glm::vec4{normalMat3[1], 0.0f};
        normalMatrix[2] = glm::vec4{normalMat3[2], 0.0f};

        materialIndex = matIndex;
    }

    [[nodiscard]] glm::mat4 getModelMatrix() const {
        return modelMatrix;
    }

    void setModelMatrix(const glm::mat4 &matrix) {
        modelMatrix = matrix;
        const glm::mat3 normalMat3 = glm::transpose(glm::inverse(glm::mat3{matrix}));
        normalMatrix[0] = glm::vec4{normalMat3[0], 0.0f};
        normalMatrix[1] = glm::vec4{normalMat3[1], 0.0f};
        normalMatrix[2] = glm::vec4{normalMat3[2], 0.0f};
    }

    [[nodiscard]] glm::mat3 getNormalMatrix() const {
        return {
            glm::vec3{normalMatrix[0]},
            glm::vec3{normalMatrix[1]},
            glm::vec3{normalMatrix[2]}
        };
    }

    [[nodiscard]] static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription{
            1, // binding (binding 1 for instance data)
            sizeof(InstanceData), // stride
            vk::VertexInputRate::eInstance // inputRate
        };
        return bindingDescription;
    }

    [[nodiscard]] static std::array<vk::VertexInputAttributeDescription, 7> getAttributeDescriptions() {
        constexpr std::uint32_t modelBase = offsetof(InstanceData, modelMatrix);
        constexpr std::uint32_t normalBase = offsetof(InstanceData, normalMatrix);
        constexpr std::uint32_t vec4Size = sizeof(glm::vec4);

        return {
            // Model matrix columns (locations 4-7)
            vk::VertexInputAttributeDescription{4, 1, vk::Format::eR32G32B32A32Sfloat, modelBase + 0u * vec4Size},
            vk::VertexInputAttributeDescription{5, 1, vk::Format::eR32G32B32A32Sfloat, modelBase + 1u * vec4Size},
            vk::VertexInputAttributeDescription{6, 1, vk::Format::eR32G32B32A32Sfloat, modelBase + 2u * vec4Size},
            vk::VertexInputAttributeDescription{7, 1, vk::Format::eR32G32B32A32Sfloat, modelBase + 3u * vec4Size},
            // Normal matrix columns (locations 8-10)
            vk::VertexInputAttributeDescription{8, 1, vk::Format::eR32G32B32A32Sfloat, normalBase + 0u * vec4Size},
            vk::VertexInputAttributeDescription{9, 1, vk::Format::eR32G32B32A32Sfloat, normalBase + 1u * vec4Size},
            vk::VertexInputAttributeDescription{10, 1, vk::Format::eR32G32B32A32Sfloat, normalBase + 2u * vec4Size}
        };
    }
};

#endif // MARBLEGAME_INSTANCE_DATA_HPP
