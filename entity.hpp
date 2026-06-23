#ifndef MARBLE_ENTITY_HPP
#define MARBLE_ENTITY_HPP

#include <memory>
#include <vector>

#include "component.hpp"

// An actor that owns a heterogeneous bag of Components.
// unique_ptr makes copy deleted and move generated automatically — no boilerplate needed.
class Entity {
public:
    // Construct a T in-place, forwarding args to its constructor.
    // Returns a non-owning pointer to the new component.
    template<typename T, typename... Args>
        requires std::derived_from<T, Component>
    [[nodiscard]] T *addComponent(Args &&... args) {
        auto &ref = m_components.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        return static_cast<T *>(ref.get());
    }

    // Returns the first component of type T, or nullptr if absent.
    template<typename T>
        requires std::derived_from<T, Component>
    [[nodiscard]] T *getComponent() const noexcept {
        for (const auto &c: m_components) {
            if (auto *p = dynamic_cast<T *>(c.get())) {
                return p;
            }
        }
        return nullptr;
    }

    // Tick every component in insertion order.
    void update(const float dt) const {
        for (const auto &c: m_components) {
            c->update(dt);
        }
    }

private:
    std::vector<std::unique_ptr<Component> > m_components;
};

#endif // MARBLE_ENTITY_HPP
