#ifndef MARBLEGAME_CAMERA_COMPONENT_HPP
#define MARBLEGAME_CAMERA_COMPONENT_HPP

#include <string_view>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "component.hpp"
#include "entity.hpp"
#include "transform_component.hpp"

class CameraComponent : public Component {
public:
    enum class ProjectionType {
        Perspective,
        Orthographic
    };

private:
    ProjectionType projectionType = ProjectionType::Perspective;

    // Perspective projection parameters
    float fieldOfView = 45.0f;
    float aspectRatio = 16.0f / 9.0f;

    // Orthographic projection parameters
    float orthoWidth = 10.0f;
    float orthoHeight = 10.0f;

    // Common parameters
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    // Camera properties
    glm::vec3 target = {0.0f, 0.0f, 0.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};

    mutable glm::mat4 viewMatrix = glm::mat4(1.0f);
    mutable glm::mat4 projectionMatrix = glm::mat4(1.0f);
    mutable bool viewMatrixDirty = true;
    mutable bool projectionMatrixDirty = true;

public:
    explicit CameraComponent(const std::string_view componentName = "CameraComponent")
        : Component{componentName} {
    }

    void initialize() override {
        updateViewMatrix();
        updateProjectionMatrix();
    }

    void setProjectionType(const ProjectionType type) {
        projectionType = type;
        projectionMatrixDirty = true;
    }

    [[nodiscard]] ProjectionType getProjectionType() const {
        return projectionType;
    }

    void setFieldOfView(const float fov) {
        fieldOfView = fov;
        projectionMatrixDirty = true;
    }

    [[nodiscard]] float getFieldOfView() const {
        return fieldOfView;
    }

    void setAspectRatio(const float ratio) {
        aspectRatio = ratio;
        projectionMatrixDirty = true;
    }

    [[nodiscard]] float getAspectRatio() const {
        return aspectRatio;
    }

    void setOrthographicSize(const float width, const float height) {
        orthoWidth = width;
        orthoHeight = height;
        projectionMatrixDirty = true;
    }

    void setClipPlanes(const float near, const float far) {
        nearPlane = near;
        farPlane = far;
        projectionMatrixDirty = true;
    }

    [[nodiscard]] float getNearPlane() const {
        return nearPlane;
    }

    [[nodiscard]] float getFarPlane() const {
        return farPlane;
    }

    void setTarget(const glm::vec3 &newTarget) {
        target = newTarget;
        viewMatrixDirty = true;
    }

    void setUp(const glm::vec3 &newUp) {
        up = newUp;
        viewMatrixDirty = true;
    }

    void lookAt(const glm::vec3 &targetPosition, const glm::vec3 &upVector = glm::vec3(0.0f, 1.0f, 0.0f)) {
        target = targetPosition;
        up = upVector;
        viewMatrixDirty = true;
    }

    [[nodiscard]] const glm::mat4 &getViewMatrix() const {
        if (viewMatrixDirty) {
            updateViewMatrix();
        }
        return viewMatrix;
    }

    [[nodiscard]] const glm::mat4 &getProjectionMatrix() const {
        if (projectionMatrixDirty) {
            updateProjectionMatrix();
        }
        return projectionMatrix;
    }

    [[nodiscard]] glm::vec3 getPosition() const {
        const auto *transform = getOwner()->getComponent<TransformComponent>();
        return transform ? transform->getPosition() : glm::vec3(0.0f, 0.0f, 0.0f);
    }

    [[nodiscard]] const glm::vec3 &getTarget() const {
        return target;
    }

    [[nodiscard]] const glm::vec3 &getUp() const {
        return up;
    }

    void forceViewMatrixUpdate() const {
        viewMatrixDirty = true;
    }

private:
    void updateViewMatrix() const {
        if (const auto *transformComponent = getOwner()->getComponent<TransformComponent>()) {
            // Build camera world transform (T * R) from the camera entity's transform
            // and compute the view matrix as its inverse. This ensures consistency
            // with rasterization and avoids relying on an external target vector.
            const glm::vec3 position = transformComponent->getPosition();
            const glm::vec3 euler = transformComponent->getRotation(); // radians

            const glm::quat qx = glm::angleAxis(euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
            const glm::quat qy = glm::angleAxis(euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
            const glm::quat qz = glm::angleAxis(euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
            const glm::quat q = qz * qy * qx; // match TransformComponent's ZYX composition

            const glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
            const glm::mat4 R = glm::mat4_cast(q);
            const glm::mat4 worldNoScale = T * R;

            viewMatrix = glm::inverse(worldNoScale);
        } else {
            // Fallback: default camera at origin looking towards +Z with Y up
            // Note: keep consistent with right-handed convention used elsewhere
            constexpr glm::vec3 position(0.0f);
            constexpr glm::vec3 forward(0.0f, 0.0f, 1.0f);
            constexpr glm::vec3 upVec(0.0f, 1.0f, 0.0f);
            viewMatrix = glm::lookAt(position, position + forward, upVec);
        }
        viewMatrixDirty = false;
    }

    void updateProjectionMatrix() const {
        if (projectionType == ProjectionType::Perspective) {
            projectionMatrix = glm::perspective(glm::radians(fieldOfView), aspectRatio, nearPlane, farPlane);
        } else {
            const float halfWidth = orthoWidth * 0.5f;
            const float halfHeight = orthoHeight * 0.5f;
            projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);
        }
        projectionMatrixDirty = false;
    }
};

#endif // MARBLEGAME_CAMERA_COMPONENT_HPP
