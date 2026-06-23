#ifndef MARBLEGAME_ENTITY_HPP
#define MARBLEGAME_ENTITY_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "component.hpp"

class Entity {
private:
    std::string name;
    bool active = true;

    std::vector<std::unique_ptr<Component> > components;

public:
    explicit Entity(const std::string_view entityName) : name{entityName} {
    }

    virtual ~Entity() = default;

    [[nodiscard]] std::string_view getName() const {
        return name;
    }

    [[nodiscard]] bool isActive() const {
        return active;
    }

    void setActive(const bool isActiveState) {
        active = isActiveState;
    }


    void initialize() const {
        for (const auto &component: components) {
            component->initialize();
        }
    }

    void update(const float deltaTime) const {
        if (!active) {
            return;
        }

        for (const auto &component: components) {
            if (component->isActive()) {
                component->update(deltaTime);
            }
        }
    }

    void render() const {
        if (!active) {
            return;
        }

        for (const auto &component: components) {
            if (component->isActive()) {
                component->render();
            }
        }
    }

    template<typename T, typename... Args>
        requires std::derived_from<T, Component>
    T *addComponent(Args &&... args) {
        auto component = std::make_unique<T>(std::forward<Args>(args)...);
        T *componentPtr = component.get();

        componentPtr->setOwner(this);

        components.push_back(std::move(component));

        componentPtr->initialize();

        return componentPtr;
    }

    template<typename T>
        requires std::derived_from<T, Component>
    [[nodiscard]] T *getComponent() const {
        for (auto it = components.rbegin(); it != components.rend(); ++it) {
            if (auto *casted = dynamic_cast<T *>(it->get())) {
                return casted;
            }
        }
        return nullptr;
    }

    template<typename T>
        requires std::derived_from<T, Component>
    bool removeComponent() {
        for (auto it = components.rbegin(); it != components.rend(); ++it) {
            if (dynamic_cast<T *>(it->get()) != nullptr) {
                components.erase(std::next(it).base());
                return true;
            }
        }
        return false;
    }

    template<typename T>
        requires std::derived_from<T, Component>
    [[nodiscard]] bool hasComponent() const {
        return getComponent<T>() != nullptr;
    }
};

#endif // MARBLEGAME_ENTITY_HPP
