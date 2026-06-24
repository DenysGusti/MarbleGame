#ifndef MARBLEGAME_PLATFORM_COMPONENT_HPP
#define MARBLEGAME_PLATFORM_COMPONENT_HPP

#include "component.hpp"

/// Holds platform-specific metadata.
/// Geometry (position, rotation, size) is stored in the sibling TransformComponent,
/// where scale == size.
class PlatformComponent final : public Component {
private:
    bool goal = false;

public:
    explicit PlatformComponent(const bool isGoal = false)
        : Component{"PlatformComponent"}, goal{isGoal} {
    }

    [[nodiscard]] bool isGoal() const { return goal; }
    void setGoal(const bool g) { goal = g; }
};

#endif // MARBLEGAME_PLATFORM_COMPONENT_HPP
