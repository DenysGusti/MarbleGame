#ifndef MARBLEGAME_TRANSFORM_COMPONENT_HPP
#define MARBLEGAME_TRANSFORM_COMPONENT_HPP

#include <string_view>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "component.hpp"

class TransformComponent final : public Component {
private:
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f}; // Euler angles in radians
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};

    // Mutable allows these cached variables to be modified inside const getter methods
    mutable glm::mat4 modelMatrix = glm::mat4(1.0f);
    mutable bool matrixDirty = true;

public:
    explicit TransformComponent(const std::string_view componentName = "TransformComponent")
        : Component{componentName} {
    }

    void setPosition(const glm::vec3 &newPosition) {
        position = newPosition;
        matrixDirty = true;
    }

    [[nodiscard]] const glm::vec3 &getPosition() const {
        return position;
    }

    void setRotation(const glm::vec3 &newRotation) {
        rotation = newRotation;
        matrixDirty = true;
    }

    [[nodiscard]] const glm::vec3 &getRotation() const {
        return rotation;
    }

    void setScale(const glm::vec3 &newScale) {
        scale = newScale;
        matrixDirty = true;
    }

    [[nodiscard]] const glm::vec3 &getScale() const {
        return scale;
    }

    void setUniformScale(const float uniformScale) {
        scale = glm::vec3(uniformScale);
        matrixDirty = true;
    }

    void translate(const glm::vec3 &translation) {
        position += translation;
        matrixDirty = true;
    }

    void rotate(const glm::vec3 &eulerAngles) {
        rotation += eulerAngles;
        matrixDirty = true;
    }

    void scaleBy(const glm::vec3 &scaleFactors) {
        scale *= scaleFactors;
        matrixDirty = true;
    }

    [[nodiscard]] const glm::mat4 &getModelMatrix() const {
        if (matrixDirty) {
            updateModelMatrix();
        }
        return modelMatrix;
    }

private:
    void updateModelMatrix() const {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), position);

        // Compose rotation with quaternions for stability and to avoid rad/deg ambiguity
        const glm::quat qx = glm::angleAxis(rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        const glm::quat qy = glm::angleAxis(rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::quat qz = glm::angleAxis(rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));

        // ZYX order is conventional for Euler composition
        const glm::quat q = qz * qy * qx;
        const glm::mat4 R = glm::mat4_cast(q);

        const glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);

        modelMatrix = T * R * S;
        matrixDirty = false;
    }
};

#endif // MARBLEGAME_TRANSFORM_COMPONENT_HPP
