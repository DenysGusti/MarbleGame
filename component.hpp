#ifndef MARBLEGAME_COMPONENT_HPP
#define MARBLEGAME_COMPONENT_HPP

#include <string>
#include <string_view>

class Entity;

class Component {
protected:
    Entity *owner = nullptr;
    std::string name;
    bool active = true;

public:
    explicit Component(const std::string_view componentName = "Component") : name(componentName) {
    }

    virtual ~Component() = default;

    virtual void initialize() {
    }

    virtual void update(const float deltaTime) {
    }

    virtual void render() const {
    }

    void setOwner(Entity *targetEntity) {
        owner = targetEntity;
    }

    [[nodiscard]] Entity *getOwner() const {
        return owner;
    }

    [[nodiscard]] std::string_view getName() const {
        return name;
    }

    [[nodiscard]] bool isActive() const {
        return active;
    }

    void setActive(const bool isActiveState) {
        active = isActiveState;
    }
};

#endif // MARBLEGAME_COMPONENT_HPP
