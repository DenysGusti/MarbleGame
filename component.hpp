#ifndef MARBLE_COMPONENT_HPP
#define MARBLE_COMPONENT_HPP

// Abstract base for every component attached to an Entity.
// Derive from this and override update() to implement behaviour.
class Component {
public:
    virtual ~Component() = default;

    // Called once per game-logic tick. dt is elapsed seconds.
    virtual void update(const float dt) {
    }
};

#endif // MARBLE_COMPONENT_HPP
