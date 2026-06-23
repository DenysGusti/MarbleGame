/* Copyright (c) 2025 Holochip Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MARBLEGAME_MESH_COMPONENT_HPP
#define MARBLEGAME_MESH_COMPONENT_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>
#include "component.hpp"
#include "instance_data.hpp"
#include "vertex.hpp"

// Forward declaration prevents circular inclusion with the heavy Model asset logic
class Model;

class MeshComponent final : public Component {
private:
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    // Cached local-space AABB (encompassing all instances)
    glm::vec3 localAABBMin{0.0f};
    glm::vec3 localAABBMax{0.0f};
    bool localAABBValid = false;

    // Cached base mesh AABB (vertices only)
    glm::vec3 meshAABBMin{0.0f};
    glm::vec3 meshAABBMax{0.0f};
    bool meshAABBValid = false;

    // All PBR texture paths for this mesh
    std::string texturePath; // Primary texture path (baseColor)
    std::string baseColorTexturePath; // Base color (albedo) texture
    std::string normalTexturePath; // Normal map texture
    std::string metallicRoughnessTexturePath; // Metallic-roughness texture
    std::string occlusionTexturePath; // Ambient occlusion texture
    std::string emissiveTexturePath; // Emissive texture

    // Instancing support
    std::vector<InstanceData> instances;

public:
    explicit MeshComponent(const std::string_view componentName = "MeshComponent")
        : Component(componentName) {
    }

    void recomputeLocalAABB() {
        recomputeMeshAABB();

        if (!meshAABBValid) {
            localAABBMin = glm::vec3(0.0f);
            localAABBMax = glm::vec3(0.0f);
            localAABBValid = false;
            return;
        }

        if (instances.empty()) {
            localAABBMin = meshAABBMin;
            localAABBMax = meshAABBMax;
            localAABBValid = true;
        } else {
            glm::vec3 fullMin(std::numeric_limits<float>::max());
            glm::vec3 fullMax(-std::numeric_limits<float>::max());

            for (const auto &inst: instances) {
                glm::vec3 instMin;
                glm::vec3 instMax;
                transformAABBLocal(inst.modelMatrix, meshAABBMin, meshAABBMax, instMin, instMax);
                fullMin = glm::min(fullMin, instMin);
                fullMax = glm::max(fullMax, instMax);
            }

            localAABBMin = fullMin;
            localAABBMax = fullMax;
            localAABBValid = true;
        }
    }

    void recomputeMeshAABB() {
        if (meshAABBValid) {
            return;
        }

        if (vertices.empty()) {
            meshAABBMin = glm::vec3(0.0f);
            meshAABBMax = glm::vec3(0.0f);
            meshAABBValid = false;
            return;
        }

        glm::vec3 minB = vertices[0].position;
        glm::vec3 maxB = vertices[0].position;
        for (const auto &v: vertices) {
            minB = glm::min(minB, v.position);
            maxB = glm::max(maxB, v.position);
        }

        meshAABBMin = minB;
        meshAABBMax = maxB;
        meshAABBValid = true;
    }

    [[nodiscard]] bool hasLocalAABB() const { return localAABBValid; }
    [[nodiscard]] glm::vec3 getLocalAABBMin() const { return localAABBMin; }
    [[nodiscard]] glm::vec3 getLocalAABBMax() const { return localAABBMax; }
    [[nodiscard]] glm::vec3 getBaseMeshAABBMin() const { return meshAABBMin; }
    [[nodiscard]] glm::vec3 getBaseMeshAABBMax() const { return meshAABBMax; }

    void setVertices(std::span<const Vertex> newVertices) {
        vertices.assign(newVertices.begin(), newVertices.end());
        meshAABBValid = false;
        localAABBValid = false;
        recomputeLocalAABB();
    }

    [[nodiscard]] std::span<const Vertex> getVertices() const {
        return vertices;
    }

    void setIndices(std::span<const std::uint32_t> newIndices) {
        indices.assign(newIndices.begin(), newIndices.end());
    }

    [[nodiscard]] std::span<const std::uint32_t> getIndices() const {
        return indices;
    }

    void setTexturePath(const std::string_view path) {
        texturePath = path;
        baseColorTexturePath = path;
    }

    [[nodiscard]] std::string_view getTexturePath() const { return texturePath; }

    void setBaseColorTexturePath(const std::string_view path) { baseColorTexturePath = path; }
    void setNormalTexturePath(const std::string_view path) { normalTexturePath = path; }
    void setMetallicRoughnessTexturePath(const std::string_view path) { metallicRoughnessTexturePath = path; }
    void setOcclusionTexturePath(const std::string_view path) { occlusionTexturePath = path; }
    void setEmissiveTexturePath(const std::string_view path) { emissiveTexturePath = path; }

    [[nodiscard]] std::string_view getBaseColorTexturePath() const { return baseColorTexturePath; }
    [[nodiscard]] std::string_view getNormalTexturePath() const { return normalTexturePath; }
    [[nodiscard]] std::string_view getMetallicRoughnessTexturePath() const { return metallicRoughnessTexturePath; }
    [[nodiscard]] std::string_view getOcclusionTexturePath() const { return occlusionTexturePath; }
    [[nodiscard]] std::string_view getEmissiveTexturePath() const { return emissiveTexturePath; }


    void createSphere(const float radius = 1.0f, const int32_t segments = 16) {
        vertices.clear();
        indices.clear();

        for (std::int32_t lat = 0; lat <= segments; ++lat) {
            const auto theta = static_cast<float>(lat * std::numbers::pi_v<double> / segments);
            const float sinTheta = std::sin(theta);
            const float cosTheta = std::cos(theta);

            for (std::int32_t lon = 0; lon <= segments; ++lon) {
                const auto phi = static_cast<float>(lon * 2.0 * std::numbers::pi_v<double> / segments);
                const float sinPhi = std::sin(phi);
                const float cosPhi = std::cos(phi);

                glm::vec3 position = {
                    radius * sinTheta * cosPhi,
                    radius * cosTheta,
                    radius * sinTheta * sinPhi
                };

                glm::vec3 normal = glm::normalize(position);
                const glm::vec2 texCoord = {
                    static_cast<float>(lon) / static_cast<float>(segments),
                    static_cast<float>(lat) / static_cast<float>(segments)
                };

                glm::vec3 tangent = {-sinTheta * sinPhi, 0.0f, sinTheta * cosPhi};
                const float len2 = glm::dot(tangent, tangent);

                if (len2 < 1e-12f) {
                    glm::vec3 t = glm::cross(normal, glm::vec3(0.0f, 0.0f, 1.0f));
                    if (glm::length(t) < 1e-12f) {
                        t = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
                    }
                    tangent = glm::normalize(t);
                } else {
                    tangent = glm::normalize(tangent);
                }

                vertices.push_back({position, normal, texCoord, glm::vec4(tangent, 1.0f)});
            }
        }

        for (std::int32_t lat = 0; lat < segments; ++lat) {
            for (std::int32_t lon = 0; lon < segments; ++lon) {
                const std::int32_t current = lat * (segments + 1) + lon;
                const std::int32_t next = current + segments + 1;

                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(current + 1);

                indices.push_back(current + 1);
                indices.push_back(next);
                indices.push_back(next + 1);
            }
        }

        recomputeLocalAABB();
    }

    void loadFromModel(const Model *model);

    void addInstance(const glm::mat4 &transform, const std::uint32_t materialIndex = 0) {
        instances.emplace_back(transform, materialIndex);
        recomputeLocalAABB();
    }

    void setInstances(std::span<const InstanceData> newInstances) {
        instances.assign(newInstances.begin(), newInstances.end());
        recomputeLocalAABB();
    }

    [[nodiscard]] std::span<const InstanceData> getInstances() const {
        return instances;
    }

    [[nodiscard]] std::size_t getInstanceCount() const {
        return instances.size();
    }

    [[nodiscard]] bool isInstanced() const {
        return instances.size() > 1;
    }

    void clearInstances() {
        instances.clear();
        recomputeLocalAABB();
    }

    void updateInstance(const std::size_t index, const glm::mat4 &transform, const std::uint32_t materialIndex = 0) {
        if (index < instances.size()) {
            instances[index] = InstanceData(transform, materialIndex);
            recomputeLocalAABB();
        }
    }

    [[nodiscard]] const InstanceData &getInstance(const std::size_t index) const {
        if (index < instances.size()) {
            return instances[index];
        }
        static const InstanceData defaultInstance;
        return instances.empty() ? defaultInstance : instances[0];
    }

private:
    static void transformAABBLocal(const glm::mat4 &M,
                                   const glm::vec3 &localMin,
                                   const glm::vec3 &localMax,
                                   glm::vec3 &outMin,
                                   glm::vec3 &outMax) {
        const glm::vec3 c = 0.5f * (localMin + localMax);
        const glm::vec3 e = 0.5f * (localMax - localMin);

        const auto worldCenter = glm::vec3(M * glm::vec4(c, 1.0f));
        const auto A = glm::mat3(M);
        const auto AbsA = glm::mat3(glm::abs(A[0]), glm::abs(A[1]), glm::abs(A[2]));
        const glm::vec3 worldExtents = AbsA * e;

        outMin = worldCenter - worldExtents;
        outMax = worldCenter + worldExtents;
    }
};

#endif // MARBLEGAME_MESH_COMPONENT_HPP
