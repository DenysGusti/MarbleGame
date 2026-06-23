#ifndef MARBLEGAME_MATERIAL_HPP
#define MARBLEGAME_MATERIAL_HPP

#include <string>
#include <string_view>
#include <glm/glm.hpp>

class Material {
private:
    std::string name;

public:
    explicit Material(const std::string_view materialName) : name{materialName} {
    }

    ~Material() = default;

    [[nodiscard]] std::string_view getName() const {
        return name;
    }

    // PBR properties (Metallic-Roughness default)
    glm::vec3 albedo = glm::vec3(1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float ao = 1.0f;
    glm::vec3 emissive = glm::vec3(0.0f);
    float ior = 1.5f; // Index of refraction
    float emissiveStrength = 1.0f; // KHR_materials_emissive_strength extension
    float alpha = 1.0f; // Base color alpha
    float transmissionFactor = 0.0f; // KHR_materials_transmission: 0=opaque, 1=fully transmissive

    // Specular-Glossiness workflow (KHR_materials_pbrSpecularGlossiness)
    bool useSpecularGlossiness = false;
    glm::vec3 specularFactor = glm::vec3(0.04f);
    float glossinessFactor = 1.0f;
    std::string specGlossTexturePath;

    // Alpha handling (glTF alphaMode and cutoff)
    std::string alphaMode = "OPAQUE"; // "OPAQUE", "MASK", or "BLEND"
    float alphaCutoff = 0.5f; // Used when alphaMode == MASK

    // Texture paths for PBR materials
    std::string albedoTexturePath;
    std::string normalTexturePath;
    std::string metallicRoughnessTexturePath;
    std::string occlusionTexturePath;
    std::string emissiveTexturePath;

    // Rendering Hints
    bool isGlass = false;
    bool isLiquid = false;
};

#endif // MARBLEGAME_MATERIAL_HPP
