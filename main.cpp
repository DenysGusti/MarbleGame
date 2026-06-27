#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
#include <vulkan/vulkan_raii.hpp>
#include <numbers>
#include <vulkan/vulkan_profiles.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include <limits>
#include <chrono>
#include <fstream>
#include <string_view>
#include <utility>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "entity.hpp"
#include "transform_component.hpp"
#include "camera_component.hpp"
#include "platform_component.hpp"
#include "vertex.hpp"

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;
constexpr std::int32_t MAX_FRAMES_IN_FLIGHT = 2;


struct GeometryData {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

class MarbleGame {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    void cleanup() {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        cleanupSwapChain();

        // Unmap memory
        for (std::size_t i = 0; i < uniformBuffersMemory.size(); i++) {
            if (uniformBuffersMapped[i] != nullptr) {
                uniformBuffersMemory[i].unmapMemory();
            }
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }

private:
    GLFWwindow *window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    std::uint32_t queueIndex = ~0;
    vk::raii::Queue queue = nullptr;
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline linePipeline = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::Buffer sphereVertexBuffer = nullptr;
    vk::raii::DeviceMemory sphereVertexBufferMemory = nullptr;
    std::uint32_t sphereIndexCount = 0;
    vk::raii::Buffer sphereIndexBuffer = nullptr;
    vk::raii::DeviceMemory sphereIndexBufferMemory = nullptr;

    vk::raii::Buffer lineVertexBuffer = nullptr;
    vk::raii::DeviceMemory lineVertexBufferMemory = nullptr;

    vk::raii::Buffer planeVertexBuffer = nullptr;
    vk::raii::DeviceMemory planeVertexBufferMemory = nullptr;
    std::uint32_t planeIndexCount = 0;
    vk::raii::Buffer planeIndexBuffer = nullptr;
    vk::raii::DeviceMemory planeIndexBufferMemory = nullptr;

    vk::raii::Buffer boxVertexBuffer = nullptr;
    vk::raii::DeviceMemory boxVertexBufferMemory = nullptr;
    std::uint32_t boxIndexCount = 0;
    vk::raii::Buffer boxIndexBuffer = nullptr;
    vk::raii::DeviceMemory boxIndexBufferMemory = nullptr;

    std::vector<std::unique_ptr<Entity> > platformEntities;
    bool hasWon = false;
    bool hasLost = false;
    bool lostByFalling = false; // true when loss was caused by falling off
    float timeRemaining = 20.0f; // countdown timer in seconds
    bool timerStarted = false; // true after the player makes their first move

    // --- Control and Physics State ---
    bool ballControlMode = true;
    glm::vec3 ballPos = glm::vec3{0.f, 2.f, 0.f};
    glm::vec3 ballVel = glm::vec3{0.f};
    glm::quat ballRot = glm::quat{1.f, 0.f, 0.f, 0.f};

    // --- Textures State ---
    vk::raii::Image ballTextureImage = nullptr;
    vk::raii::DeviceMemory ballTextureImageMemory = nullptr;
    vk::raii::ImageView ballTextureImageView = nullptr;

    vk::raii::Image floorTextureImage = nullptr;
    vk::raii::DeviceMemory floorTextureImageMemory = nullptr;
    vk::raii::ImageView floorTextureImageView = nullptr;

    vk::raii::Image goalTextureImage = nullptr;
    vk::raii::DeviceMemory goalTextureImageMemory = nullptr;
    vk::raii::ImageView goalTextureImageView = nullptr;

    vk::raii::Sampler textureSampler = nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSets descriptorSets = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffers commandBuffers = nullptr;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    std::uint32_t frameIndex = 0;
    bool framebufferResized = false;

    // --- Camera State for Navigation ---
    glm::vec3 cameraPos{0.f, 0.f, 4.f};
    glm::vec3 cameraFront{0.f, 0.f, -1.f};
    glm::vec3 cameraUp{0.f, 1.f, 0.f};
    float yaw = -90.f;
    float pitch = 0.f;
    double lastX = 400.0;
    double lastY = 300.0;
    bool firstMouse = true;

    // --- ECS Entity/Component State ---
    std::unique_ptr<Entity> cameraEntity;
    std::unique_ptr<Entity> ballEntity;

    void addPlatform(const glm::vec3 &position, const glm::vec3 &size, const glm::vec3 &rotation,
                     const bool isGoal = false, const std::string_view name = "") {
        auto entity = std::make_unique<Entity>(name);
        auto *t = entity->addComponent<TransformComponent>();
        t->setPosition(position);
        t->setRotation(rotation);
        t->setScale(size);
        entity->addComponent<PlatformComponent>(isGoal);
        platformEntities.push_back(std::move(entity));
    }

    void initEntities() {
        cameraEntity = std::make_unique<Entity>("Camera");
        auto *cameraTransform = cameraEntity->addComponent<TransformComponent>();
        cameraTransform->setPosition(cameraPos);
        cameraTransform->setRotation(glm::vec3{glm::radians(pitch), -glm::radians(yaw + 90.f), 0.f});

        auto *cameraComp = cameraEntity->addComponent<CameraComponent>();
        cameraComp->setFieldOfView(45.0f);
        cameraComp->setAspectRatio(
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height));
        cameraComp->setClipPlanes(0.1f, 100.0f);
        cameraComp->initialize();

        ballEntity = std::make_unique<Entity>("Ball");
        auto *ballTransform = ballEntity->addComponent<TransformComponent>();

        // Initial spawn position (High up to match the detailed map)
        ballPos = glm::vec3{0.f, 22.f, 0.f};
        ballTransform->setPosition(ballPos);
        ballTransform->setRotation(glm::eulerAngles(ballRot));

        platformEntities.clear();
        hasWon = false;
        hasLost = false;
        timeRemaining = 20.0f;

        // --- 1. START AREA ---
        addPlatform({0.f, 20.f, 0.f}, {10.f, 1.f, 10.f}, {0.f, 0.f, 0.f}, false, "Start Pad");

        // The "V" shaped slopes behind the start area
        addPlatform({-7.f, 21.5f, 0.f}, {6.f, 1.f, 10.f}, {0.f, 0.f, glm::radians(-30.f)}, false, "Start Left Slope");
        addPlatform({7.f, 21.5f, 0.f}, {6.f, 1.f, 10.f}, {0.f, 0.f, glm::radians(30.f)}, false, "Start Right Slope");

        // Initial Drop
        addPlatform({0.f, 18.5f, 8.f}, {10.f, 1.f, 7.f}, {glm::radians(25.f), 0.f, 0.f}, false, "Initial Drop");
        // Pre-Twin Ramps Landing
        addPlatform({0.f, 17.f, 13.f}, {10.f, 1.f, 4.f}, {0.f, 0.f, 0.f}, false, "Pre-Twin Landing");

        // --- 2. TWIN RAMPS WITH RAILS ---
        addPlatform({-3.f, 15.5f, 19.f}, {3.f, 1.f, 10.f}, {glm::radians(16.7f), 0.f, 0.f}, false, "Left Twin Ramp");
        addPlatform({-4.8f, 16.f, 19.f}, {0.6f, 1.5f, 10.f}, {glm::radians(16.7f), 0.f, 0.f}, false, "Left Rail");
        addPlatform({3.f, 15.5f, 19.f}, {3.f, 1.f, 10.f}, {glm::radians(16.7f), 0.f, 0.f}, false, "Right Twin Ramp");
        addPlatform({4.8f, 16.f, 19.f}, {0.6f, 1.5f, 10.f}, {glm::radians(16.7f), 0.f, 0.f}, false, "Right Rail");

        // --- 3. MIDDLE LANDING & PYRAMIDS ---
        addPlatform({0.f, 14.f, 25.f}, {12.f, 1.f, 3.f}, {0.f, 0.f, 0.f}, false, "Middle Landing Top");
        addPlatform({-3.5f, 14.f, 28.5f}, {5.f, 1.f, 4.f}, {0.f, 0.f, 0.f}, false, "Middle Bridge Left");
        addPlatform({3.5f, 14.f, 28.5f}, {5.f, 1.f, 4.f}, {0.f, 0.f, 0.f}, false, "Middle Bridge Right");
        addPlatform({-2.5f, 14.8f, 28.5f}, {2.5f, 2.5f, 2.5f}, {0.f, glm::radians(45.f), 0.f}, false, "Pyramid 1");
        addPlatform({2.5f, 14.8f, 28.5f}, {2.5f, 2.5f, 2.5f}, {0.f, glm::radians(45.f), 0.f}, false, "Pyramid 2");
        addPlatform({0.f, 14.f, 32.f}, {12.f, 1.f, 3.f}, {0.f, 0.f, 0.f}, false, "Middle Landing Bottom");

        // --- 4. STEEP CHUTE ---
        addPlatform({0.f, 10.f, 38.f}, {4.f, 1.f, 11.5f}, {glm::radians(40.f), 0.f, 0.f}, false, "Steep Chute");

        // --- 5. ZIG-ZAG SECTION ---
        addPlatform({0.f, 6.f, 45.f}, {6.f, 1.f, 6.f}, {0.f, 0.f, 0.f}, false, "Catch Pad 1");
        addPlatform({0.f, 7.5f, 48.5f}, {6.f, 4.f, 1.f}, {0.f, 0.f, 0.f}, false, "Catch Wall 1");
        addPlatform({8.f, 5.f, 45.f}, {10.f, 1.f, 4.f}, {0.f, 0.f, glm::radians(-11.3f)}, false, "Zig-Zag Right Ramp");
        addPlatform({15.f, 4.f, 45.f}, {6.f, 1.f, 6.f}, {0.f, 0.f, 0.f}, false, "Catch Pad 2");
        addPlatform({18.5f, 5.5f, 45.f}, {1.f, 4.f, 6.f}, {0.f, 0.f, 0.f}, false, "Catch Wall 2");
        addPlatform({15.f, 2.5f, 52.f}, {4.f, 1.f, 10.f}, {glm::radians(16.7f), 0.f, 0.f}, false,
                    "Zig-Zag Forward Ramp");

        // --- 6. GOAL AREA ---
        addPlatform({15.f, 1.f, 59.f}, {6.f, 1.f, 6.f}, {0.f, 0.f, 0.f}, false, "Pre-Goal Landing");
        addPlatform({15.f, 0.5f, 64.f}, {4.f, 1.f, 6.f}, {glm::radians(9.5f), 0.f, 0.f}, false, "Final Mini Ramp");
        addPlatform({15.f, 0.f, 71.f}, {12.f, 1.f, 10.f}, {0.f, 0.f, 0.f}, true, "Goal Pad");
    }

    std::vector<const char *> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

    static void framebufferResizeCallback(GLFWwindow *window, std::int32_t width, std::int32_t height) {
        const auto app = static_cast<MarbleGame *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Marble Madness", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        // Capture and disable the mouse cursor initially
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported()) {
            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        }
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createGeometryBuffers();
        createUniformBuffers();
        createTextureImage();
        createTextureSampler();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        initImgui();
        initEntities();
    }

    void initImgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);

        auto colorFormat = static_cast<VkFormat>(swapChainSurfaceFormat.format);
        const VkPipelineRenderingCreateInfo renderingInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorFormat,
            .depthAttachmentFormat = static_cast<VkFormat>(findDepthFormat())
        };

        ImGui_ImplVulkan_InitInfo initInfo = {};
        initInfo.Instance = *instance;
        initInfo.PhysicalDevice = *physicalDevice;
        initInfo.Device = *device;
        initInfo.QueueFamily = queueIndex;
        initInfo.Queue = *queue;
        initInfo.DescriptorPool = VK_NULL_HANDLE;
        initInfo.DescriptorPoolSize = 100;
        initInfo.MinImageCount = chooseSwapMinImageCount(physicalDevice.getSurfaceCapabilitiesKHR(*surface));
        initInfo.ImageCount = static_cast<std::uint32_t>(swapChainImages.size());
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        ImGui_ImplVulkan_Init(&initInfo);
    }

    void createInstance() {
        constexpr vk::ApplicationInfo vkAppInfo{
            .pApplicationName = "MarbleGame",
            .applicationVersion = 1,
            .pEngineName = "Custom Engine",
            .engineVersion = 1,
            .apiVersion = vk::ApiVersion13
        };

        auto extensions = getRequiredInstanceExtensions();

        const vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &vkAppInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        instance = vk::raii::Instance{context, createInfo};
    }

    void createSurface() {
        VkSurfaceKHR rawSurface = nullptr;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error{"failed to create window surface!"};
        }
        surface = vk::raii::SurfaceKHR{instance, rawSurface};
    }

    bool isDeviceSuitable(vk::raii::PhysicalDevice const &physDevice) {
        bool supportsVulkan1_3 = physDevice.getProperties().apiVersion >= vk::ApiVersion13;

        auto queueFamilies = physDevice.getQueueFamilyProperties();
        bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) {
            return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

        auto availableDeviceExtensions = physDevice.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions =
                std::ranges::all_of(requiredDeviceExtension,
                                    [&availableDeviceExtensions](auto const &requiredExt) {
                                        return std::ranges::any_of(availableDeviceExtensions,
                                                                   [requiredExt](auto const &availExt) {
                                                                       return std::string_view(availExt.extensionName)
                                                                              == requiredExt;
                                                                   });
                                    });

        auto features = physDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool supportsRequiredFeatures = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().
                                        extendedDynamicState;

        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
        auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const &physDevice) {
            return isDeviceSuitable(physDevice);
        });
        if (devIter == physicalDevices.end()) {
            throw std::runtime_error{"failed to find a suitable GPU!"};
        }
        physicalDevice = *devIter;

        constexpr VpProfileProperties profileProperties = {
            VP_KHR_ROADMAP_2022_NAME,
            VP_KHR_ROADMAP_2022_SPEC_VERSION
        };

        VkBool32 supported = vk::False;
        const VkResult vk_result = vpGetPhysicalDeviceProfileSupport(*instance, *physicalDevice, &profileProperties,
                                                                     &supported);

        if (static_cast<vk::Result>(vk_result) != vk::Result::eSuccess || supported != vk::True) {
            throw std::runtime_error{"Roadmap 2022 profile is not supported on this device"};
        }
    }

    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        for (std::uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
            if (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0) {
            throw std::runtime_error{"Could not find a queue for graphics and present -> terminating"};
        }

        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
            .queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority
        };
        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.wideLines = vk::True;
        deviceFeatures.samplerAnisotropy = vk::True;

        vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQueueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(requiredDeviceExtension.size()),
            .ppEnabledExtensionNames = requiredDeviceExtension.data(),
            .pEnabledFeatures = &deviceFeatures
        };

        VkDevice rawDevice = nullptr;
        VkDeviceCreateInfo rawCreateInfo = deviceCreateInfo;

        constexpr VpProfileProperties profileProperties = {
            VP_KHR_ROADMAP_2022_NAME,
            VP_KHR_ROADMAP_2022_SPEC_VERSION
        };

        VpDeviceCreateInfo vpCreateInfo{
            .pCreateInfo = &rawCreateInfo,
            .flags = 0,
            .enabledFullProfileCount = 1,
            .pEnabledFullProfiles = &profileProperties,
            .enabledProfileBlockCount = 0,
            .pEnabledProfileBlocks = nullptr
        };

        if (VkResult res = vpCreateDevice(*physicalDevice, &vpCreateInfo, nullptr, &rawDevice); res != VK_SUCCESS) {
            throw std::runtime_error{"failed to create logical device via Vulkan profile!"};
        }

        device = vk::raii::Device{physicalDevice, rawDevice};
        queue = vk::raii::Queue{device, queueIndex, 0};
    }

    void createSwapChain() {
        const vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        const std::uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

        const std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

        const std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.
                getSurfacePresentModesKHR(*surface);
        const vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);

        const vk::SwapchainCreateInfoKHR swapChainCreateInfo{
            .surface = *surface,
            .minImageCount = minImageCount,
            .imageFormat = swapChainSurfaceFormat.format,
            .imageColorSpace = swapChainSurfaceFormat.colorSpace,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = surfaceCapabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = presentMode,
            .clipped = true
        };

        swapChain = vk::raii::SwapchainKHR{device, swapChainCreateInfo};
        swapChainImages = swapChain.getImages();
    }

    void createImageViews() {
        assert(swapChainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainSurfaceFormat.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
        for (const auto &image: swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    void createDepthResources() {
        const vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
                    depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    [[nodiscard]] vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                                 const vk::ImageTiling tiling,
                                                 const vk::FormatFeatureFlags features) const {
        for (const auto format: candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error{"failed to find supported format!"};
    }

    void createDescriptorSetLayout() {
        constexpr std::array bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            }
        };
        const vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };
        descriptorSetLayout = vk::raii::DescriptorSetLayout{device, layoutInfo};
    }

    void createGraphicsPipeline() {
        const auto vertShaderCode = readFile("main_vert.spv");
        const auto fragShaderCode = readFile("main_frag.spv");

        const vk::ShaderModuleCreateInfo vInfo{
            .codeSize = vertShaderCode.size(),
            .pCode = reinterpret_cast<const std::uint32_t *>(vertShaderCode.data())
        };
        const vk::raii::ShaderModule vertModule{device, vInfo};

        const vk::ShaderModuleCreateInfo fInfo{
            .codeSize = fragShaderCode.size(),
            .pCode = reinterpret_cast<const std::uint32_t *>(fragShaderCode.data())
        };
        const vk::raii::ShaderModule fragModule{device, fInfo};

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertModule, .pName = "main"
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragModule, .pName = "main"
        };
        const std::array shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };
        pipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

        vk::Format depthFormat = findDepthFormat();

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {
                .stageCount = 2,
                .pStages = shaderStages.data(),
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = *pipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
                .depthAttachmentFormat = depthFormat
            }
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr,
                                              pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

        // Create line graphics pipeline
        vk::PipelineInputAssemblyStateCreateInfo lineInputAssembly{
            .topology = vk::PrimitiveTopology::eLineList,
            .primitiveRestartEnable = vk::False
        };

        vk::PipelineRasterizationStateCreateInfo lineRasterizer = rasterizer;
        lineRasterizer.cullMode = vk::CullModeFlagBits::eNone;
        lineRasterizer.lineWidth = 3.f;

        vk::PipelineDepthStencilStateCreateInfo lineDepthStencil = depthStencil;
        lineDepthStencil.depthTestEnable = vk::False;
        lineDepthStencil.depthWriteEnable = vk::False;

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> linePipelineCreateInfoChain
                = {
                    {
                        .stageCount = 2,
                        .pStages = shaderStages.data(),
                        .pVertexInputState = &vertexInputInfo,
                        .pInputAssemblyState = &lineInputAssembly,
                        .pViewportState = &viewportState,
                        .pRasterizationState = &lineRasterizer,
                        .pMultisampleState = &multisampling,
                        .pDepthStencilState = &lineDepthStencil,
                        .pColorBlendState = &colorBlending,
                        .pDynamicState = &dynamicState,
                        .layout = *pipelineLayout,
                        .renderPass = nullptr
                    },
                    {
                        .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
                        .depthAttachmentFormat = depthFormat
                    }
                };

        linePipeline = vk::raii::Pipeline(device, nullptr,
                                          linePipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void createCommandPool() {
        const vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIndex
        };
        commandPool = vk::raii::CommandPool{device, poolInfo};
    }

    void createGeometryBuffers() {
        const GeometryData sphere = createSphereGeometry(1.f, 30, 30);
        sphereIndexCount = static_cast<std::uint32_t>(sphere.indices.size());

        // Vertex Buffer Allocation and Copy
        {
            const vk::DeviceSize bufferSize = sizeof(Vertex) * sphere.vertices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                               vk::MemoryPropertyFlagBits::eHostCoherent);

            auto *dest = static_cast<Vertex *>(stagingMemory.mapMemory(0, bufferSize));
            std::ranges::copy(sphere.vertices, dest);
            stagingMemory.unmapMemory();

            std::tie(sphereVertexBuffer, sphereVertexBufferMemory) = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal);

            copyBuffer(stagingBuffer, sphereVertexBuffer, bufferSize);
        }

        // Index Buffer Allocation and Copy
        {
            const vk::DeviceSize bufferSize = sizeof(std::uint32_t) * sphere.indices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                               vk::MemoryPropertyFlagBits::eHostCoherent);

            auto *dest = static_cast<std::uint32_t *>(stagingMemory.mapMemory(0, bufferSize));
            std::ranges::copy(sphere.indices, dest);
            stagingMemory.unmapMemory();

            std::tie(sphereIndexBuffer, sphereIndexBufferMemory) = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal);

            copyBuffer(stagingBuffer, sphereIndexBuffer, bufferSize);
        }

        // Line Vertex Buffer (Static, Device-Local)
        {
            constexpr float lineLength = 0.015f;
            constexpr std::array lineVertices = {
                // X-axis (Red)
                Vertex{
                    .position = glm::vec3{0.f}, .normal = glm::vec3{0, 1, 0},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{1.f, 0.f, 0.f, 1.f}
                },
                Vertex{
                    .position = glm::vec3{lineLength, 0.f, 0.f}, .normal = glm::vec3{0, 1, 0},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{1.f, 0.f, 0.f, 1.f}
                },
                // Y-axis (Green)
                Vertex{
                    .position = glm::vec3{0.f}, .normal = glm::vec3{0, 0, 1},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{0.f, 1.f, 0.f, 1.f}
                },
                Vertex{
                    .position = glm::vec3{0.f, lineLength, 0.f}, .normal = glm::vec3{0, 0, 1},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{0.f, 1.f, 0.f, 1.f}
                },
                // Z-axis (Blue)
                Vertex{
                    .position = glm::vec3{0.f}, .normal = glm::vec3{1, 0, 0},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{0.f, 0.f, 1.f, 1.f}
                },
                Vertex{
                    .position = glm::vec3{0.f, 0.f, lineLength}, .normal = glm::vec3{1, 0, 0},
                    .texCoord = glm::vec2(0.f), .tangent = glm::vec4{0.f, 0.f, 1.f, 1.f}
                }
            };

            constexpr vk::DeviceSize bufferSize = sizeof(Vertex) * lineVertices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                               vk::MemoryPropertyFlagBits::eHostCoherent);

            auto *dest = static_cast<Vertex *>(stagingMemory.mapMemory(0, bufferSize));
            std::ranges::copy(lineVertices, dest);
            stagingMemory.unmapMemory();

            std::tie(lineVertexBuffer, lineVertexBufferMemory) = createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal);

            copyBuffer(stagingBuffer, lineVertexBuffer, bufferSize);
        }

        // Plane Geometry Allocation and Copy
        {
            float planeSize = 50.f;
            float planeY = -1.f;
            std::vector planeVertices = {
                Vertex{
                    .position = glm::vec3{-planeSize, planeY, -planeSize}, .normal = glm::vec3{0.f, 1.f, 0.f},
                    .texCoord = glm::vec2(0.f, 0.f), .tangent = glm::vec4{0.f}
                },
                Vertex{
                    .position = glm::vec3{planeSize, planeY, -planeSize}, .normal = glm::vec3{0.f, 1.f, 0.f},
                    .texCoord = glm::vec2(planeSize, 0.f), .tangent = glm::vec4{0.f}
                },
                Vertex{
                    .position = glm::vec3{planeSize, planeY, planeSize}, .normal = glm::vec3{0.f, 1.f, 0.f},
                    .texCoord = glm::vec2(planeSize, planeSize), .tangent = glm::vec4{0.f}
                },
                Vertex{
                    .position = glm::vec3{-planeSize, planeY, planeSize}, .normal = glm::vec3{0.f, 1.f, 0.f},
                    .texCoord = glm::vec2(0.f, planeSize), .tangent = glm::vec4{0.f}
                }
            };
            std::vector<std::uint32_t> planeIndices = {
                0, 2, 1,
                0, 3, 2
            };
            planeIndexCount = static_cast<std::uint32_t>(planeIndices.size());

            // Vertex Buffer
            {
                const vk::DeviceSize bufferSize = sizeof(Vertex) * planeVertices.size();
                auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
                auto *dest = static_cast<Vertex *>(stagingMemory.mapMemory(0, bufferSize));
                std::ranges::copy(planeVertices, dest);
                stagingMemory.unmapMemory();
                std::tie(planeVertexBuffer, planeVertexBufferMemory) = createBuffer(
                    bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
                copyBuffer(stagingBuffer, planeVertexBuffer, bufferSize);
            }

            // Index Buffer
            {
                const vk::DeviceSize bufferSize = sizeof(std::uint32_t) * planeIndices.size();
                auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
                auto *dest = static_cast<std::uint32_t *>(stagingMemory.mapMemory(0, bufferSize));
                std::ranges::copy(planeIndices, dest);
                stagingMemory.unmapMemory();
                std::tie(planeIndexBuffer, planeIndexBufferMemory) = createBuffer(
                    bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
                copyBuffer(stagingBuffer, planeIndexBuffer, bufferSize);
            }
        }

        // Box Geometry Allocation and Copy
        {
            const GeometryData box = createBoxGeometry();
            boxIndexCount = static_cast<std::uint32_t>(box.indices.size());

            // Box Vertex Buffer
            {
                const vk::DeviceSize bufferSize = sizeof(Vertex) * box.vertices.size();
                auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
                auto *dest = static_cast<Vertex *>(stagingMemory.mapMemory(0, bufferSize));
                std::ranges::copy(box.vertices, dest);
                stagingMemory.unmapMemory();
                std::tie(boxVertexBuffer, boxVertexBufferMemory) = createBuffer(
                    bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
                copyBuffer(stagingBuffer, boxVertexBuffer, bufferSize);
            }

            // Box Index Buffer
            {
                const vk::DeviceSize bufferSize = sizeof(std::uint32_t) * box.indices.size();
                auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
                auto *dest = static_cast<std::uint32_t *>(stagingMemory.mapMemory(0, bufferSize));
                std::ranges::copy(box.indices, dest);
                stagingMemory.unmapMemory();
                std::tie(boxIndexBuffer, boxIndexBufferMemory) = createBuffer(
                    bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
                copyBuffer(stagingBuffer, boxIndexBuffer, bufferSize);
            }
        }
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            constexpr vk::DeviceSize bufferSize = sizeof(CameraUBO);
            auto [buffer, bufferMem] = createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                                    vk::MemoryPropertyFlagBits::eHostVisible |
                                                    vk::MemoryPropertyFlagBits::eHostCoherent);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createTexture(const std::string &filepath, vk::raii::Image &textureImage,
                       vk::raii::DeviceMemory &textureImageMemory, vk::raii::ImageView &textureImageView) const {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error{"failed to load texture image: " + filepath};
        }
        const vk::DeviceSize imageSize = texWidth * texHeight * 4;

        auto [stagingBuffer, stagingMemory] = createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                           vk::MemoryPropertyFlagBits::eHostVisible |
                                                           vk::MemoryPropertyFlagBits::eHostCoherent);

        void *data = stagingMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingMemory.unmapMemory();

        stbi_image_free(pixels);

        const std::uint32_t mipLevels = static_cast<std::uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight))))
                                        + 1;

        createImage(texWidth, texHeight, mipLevels, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                    vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffer cmd = std::move(device.allocateCommandBuffers(allocInfo).front());
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        transition_image_layout(cmd, *textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                {}, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eTransfer,
                                vk::ImageAspectFlagBits::eColor, mipLevels);

        const vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1}
        };
        cmd.copyBufferToImage(*stagingBuffer, *textureImage, vk::ImageLayout::eTransferDstOptimal, region);

        cmd.end();
        queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd}, nullptr);
        queue.waitIdle();

        generateMipmaps(*textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);

        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor,
                                           mipLevels);
    }

    void generateMipmaps(const vk::Image image, const vk::Format imageFormat, const std::int32_t texWidth,
                         const std::int32_t texHeight, const std::uint32_t mipLevels) const {
        const vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error{"texture image format does not support linear blitting!"};
        }

        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffer cmd = std::move(device.allocateCommandBuffers(allocInfo).front());
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        std::int32_t mipWidth = texWidth;
        std::int32_t mipHeight = texHeight;

        for (std::uint32_t i = 1; i < mipLevels; i++) {
            vk::ImageMemoryBarrier2 barrierSrc{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = image,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = i - 1,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };
            const vk::DependencyInfo depSrc = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrierSrc
            };
            cmd.pipelineBarrier2(depSrc);

            const std::array srcOffsets = {
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{mipWidth, mipHeight, 1}
            };
            const std::array dstOffsets = {
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1}
            };
            const vk::ImageBlit blit = {
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
                .srcOffsets = srcOffsets,
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1},
                .dstOffsets = dstOffsets
            };
            cmd.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
                          {blit}, vk::Filter::eLinear);

            vk::ImageMemoryBarrier2 barrierShader{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = image,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = i - 1,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };
            const vk::DependencyInfo depShader = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrierShader
            };
            cmd.pipelineBarrier2(depShader);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        vk::ImageMemoryBarrier2 barrierFinal{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = mipLevels - 1,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        const vk::DependencyInfo depFinal = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrierFinal
        };
        cmd.pipelineBarrier2(depFinal);

        cmd.end();
        queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd}, nullptr);
        queue.waitIdle();
    }

    void createTextureImage() {
        createTexture("textures/marble.jpg", ballTextureImage, ballTextureImageMemory, ballTextureImageView);
        createTexture("textures/floor.jpg", floorTextureImage, floorTextureImageMemory, floorTextureImageView);
        createTexture("textures/goal.jpg", goalTextureImage, goalTextureImageMemory, goalTextureImageView);
    }

    void createTextureSampler() {
        const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        const vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 16.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False
        };
        textureSampler = vk::raii::Sampler{device, samplerInfo};
    }

    void createDescriptorPool() {
        std::array poolSizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * 3}
        };
        const vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        descriptorPool = vk::raii::DescriptorPool{device, poolInfo};
    }

    void createDescriptorSets() {
        std::vector layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<std::uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };
        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = *uniformBuffers[i],
                .offset = 0,
                .range = sizeof(CameraUBO)
            };
            vk::DescriptorImageInfo ballImageInfo{
                .sampler = *textureSampler,
                .imageView = *ballTextureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo floorImageInfo{
                .sampler = *textureSampler,
                .imageView = *floorTextureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo goalImageInfo{
                .sampler = *textureSampler,
                .imageView = *goalTextureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            std::array descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &ballImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &floorImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &goalImageInfo
                }
            };
            device.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createCommandBuffers() {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };
        commandBuffers = vk::raii::CommandBuffers{device, allocInfo};
    }

    void createSyncObjects() {
        assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

        for (std::size_t i = 0; i < swapChainImages.size(); i++) {
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        }

        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void recreateSwapChain() {
        std::int32_t width{0}, height{0};
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();

        renderFinishedSemaphores.clear();
        for (std::size_t i = 0; i < swapChainImages.size(); i++) {
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        }
    }

    void recordCommandBuffer(vk::raii::CommandBuffer &commandBuffer, const std::uint32_t imageIndex,
                             const std::uint32_t frameIdx) const {
        constexpr vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);

        // Transition swapchain image
        transition_image_layout(
            commandBuffer,
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        // Transition depth image
        transition_image_layout(
            commandBuffer,
            *depthImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::ImageAspectFlagBits::eDepth);

        constexpr vk::ClearValue clearColorValue{
            .color = vk::ClearColorValue{
                .float32 = std::array{0.1f, 0.2f, 0.3f, 1.f}
            }
        };

        constexpr vk::ClearValue depthClearValue{
            .depthStencil = {
                .depth = 1.f,
                .stencil = 0
            }
        };

        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eNone,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColorValue
        };

        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView = *depthImageView,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eNone,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = depthClearValue
        };

        const vk::RenderingInfo renderingInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment,
            .pDepthAttachment = &depthAttachment
        };

        commandBuffer.beginRendering(renderingInfo);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        const vk::Viewport viewport{
            .x = 0.f,
            .y = static_cast<float>(swapChainExtent.height),
            .width = static_cast<float>(swapChainExtent.width),
            .height = -static_cast<float>(swapChainExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f
        };
        commandBuffer.setViewport(0, viewport);

        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = swapChainExtent
        };
        commandBuffer.setScissor(0, scissor);

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout,
            0,
            {*descriptorSets[frameIdx]},
            nullptr
        );

        // Draw the platforms
        for (const auto &ent: platformEntities) {
            const auto *t = ent->getComponent<TransformComponent>();
            const auto *p = ent->getComponent<PlatformComponent>();
            if (!t || !p) continue;

            PushConstants pcs{
                .model = t->getModelMatrix(),
                .size = t->getScale(),
                .textureIndex = p->isGoal() ? TextureIndex::Goal : TextureIndex::Floor,
            };
            commandBuffer.pushConstants<PushConstants>(*pipelineLayout,
                                                       vk::ShaderStageFlagBits::eVertex |
                                                       vk::ShaderStageFlagBits::eFragment, 0, pcs);
            commandBuffer.bindVertexBuffers(0, {*boxVertexBuffer}, {0});
            commandBuffer.bindIndexBuffer(*boxIndexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(boxIndexCount, 1, 0, 0, 0);
        }

        // Draw the sphere
        {
            glm::mat4 sphereModel;
            if (ballEntity) {
                if (auto *ballTransform = ballEntity->getComponent<TransformComponent>()) {
                    sphereModel = ballTransform->getModelMatrix();
                } else {
                    sphereModel = glm::translate(glm::mat4{1.0f}, ballPos) * glm::mat4_cast(ballRot);
                }
            } else {
                sphereModel = glm::translate(glm::mat4{1.0f}, ballPos) * glm::mat4_cast(ballRot);
            }
            PushConstants pcs{
                .model = sphereModel,
                .size = glm::vec3{1.0f, 1.0f, 1.0f},
                .textureIndex = TextureIndex::Ball,
            };
            commandBuffer.pushConstants<PushConstants>(*pipelineLayout,
                                                       vk::ShaderStageFlagBits::eVertex |
                                                       vk::ShaderStageFlagBits::eFragment, 0, pcs);
            commandBuffer.bindVertexBuffers(0, {*sphereVertexBuffer}, {0});
            commandBuffer.bindIndexBuffer(*sphereIndexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(sphereIndexCount, 1, 0, 0, 0);
        }

        // Draw debug lines (only in Fly Mode)
        if (!ballControlMode) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *linePipeline);
            auto identity = glm::mat4{1.0f};
            PushConstants pcs{
                .model = identity,
                .textureIndex = TextureIndex::None
            };
            commandBuffer.pushConstants<PushConstants>(*pipelineLayout,
                                                       vk::ShaderStageFlagBits::eVertex |
                                                       vk::ShaderStageFlagBits::eFragment, 0, pcs);
            commandBuffer.bindVertexBuffers(0, {*lineVertexBuffer}, {0});
            commandBuffer.draw(6, 1, 0, 0);
        }

        // ImGui overlay
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);

        commandBuffer.endRendering();

        // Transition swapchain image to PRESENT_SRC
        transition_image_layout(
            commandBuffer,
            swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor);

        commandBuffer.end();
    }

    void drawFrame() {
        const auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True,
                                                      std::numeric_limits<std::uint64_t>::max());
        if (fenceResult != vk::Result::eSuccess) {
            throw std::runtime_error{"failed to wait for fence!"};
        }

        auto [result, imageIndex] = swapChain.acquireNextImage(std::numeric_limits<std::uint64_t>::max(),
                                                               *presentCompleteSemaphores[frameIndex], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error{"failed to acquire swap chain image!"};
        }

        // ImGui Frame Setup
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float displayYaw = std::fmod(yaw, 360.f);
        if (displayYaw < 0.f) {
            displayYaw += 360.f;
        }

        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("HUD", nullptr,
                     ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav);

        if (!ballControlMode) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.f, 1.f), "MODE: FLY MODE [TAB to switch]");
            ImGui::Text("Camera Pos: %.3f / %.3f / %.3f", cameraPos.x, cameraPos.y, cameraPos.z);
            ImGui::Text("Controls: WASD=Fly, Mouse=Look, Space/LShift=Up/Down");
        } else {
            ImGui::TextColored(ImVec4(0.3f, 1.f, 0.4f, 1.f), "MODE: BALL CONTROL MODE [TAB to switch]");
            ImGui::Text("Ball Pos: %.3f / %.3f / %.3f", ballPos.x, ballPos.y, ballPos.z);
            ImGui::Text("Ball Speed: %.2f m/s", glm::length(glm::vec3{ballVel.x, 0.f, ballVel.z}));
            ImGui::Text("Controls: WASD=Roll Ball, Mouse=Orbit Camera, Space=Jump, R=Restart");
        }
        // Timer always visible once started
        if (timerStarted) {
            if (timeRemaining <= 5.f)
                ImGui::TextColored(ImVec4(1.f, 0.15f, 0.15f, 1.f), "TIME: %.1f s", timeRemaining);
            else
                ImGui::TextColored(ImVec4(1.f, 0.85f, 0.1f, 1.f), "TIME: %.1f s", timeRemaining);
        }
        ImGui::Text("Facing: %.1f / %.1f (Yaw / Pitch)", displayYaw, pitch);
        ImGui::End();

        if (hasWon) {
            ImGui::SetNextWindowPos(ImVec2(static_cast<float>(swapChainExtent.width) * 0.5f - 150.f,
                                           static_cast<float>(swapChainExtent.height) * 0.5f - 110.f),
                                    ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(300.f, 220.f));
            ImGui::Begin("Victory!", nullptr,
                         ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove);
            ImGui::TextColored(ImVec4(0.1f, 1.f, 0.1f, 1.f), "YOU REACHED THE GOAL!");
            ImGui::Separator();
            ImGui::Text("Nice roll!");
            ImGui::Text("Time left: %.1f s", timeRemaining);
            ImGui::Spacing();
            if (ImGui::Button("Play Again [R]", ImVec2(284.f, 40.f))) {
                ballPos = glm::vec3{0.f, 22.f, 0.f};
                ballVel = glm::vec3{0.f};
                ballRot = glm::quat{1.f, 0.f, 0.f, 0.f};
                hasWon = false;
                hasLost = false;
                lostByFalling = false;
                timeRemaining = 20.0f;
                timerStarted = false;
            }
            ImGui::End();
        }

        if (hasLost) {
            ImGui::SetNextWindowPos(ImVec2(static_cast<float>(swapChainExtent.width) * 0.5f - 150.f,
                                           static_cast<float>(swapChainExtent.height) * 0.5f - 110.f),
                                    ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(300.f, 220.f));
            ImGui::Begin(lostByFalling ? "You Fell!" : "Time's Up!", nullptr,
                         ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove);
            ImGui::TextColored(ImVec4(1.f, 0.15f, 0.15f, 1.f),
                               lostByFalling ? "YOU FELL OFF! YOU LOSE!" : "TIME'S UP! YOU LOSE!");
            ImGui::Separator();
            ImGui::Text(lostByFalling ? "Don't fall off the edge!" : "You ran out of time.");
            ImGui::Text("Press Play Again to retry.");
            ImGui::Spacing();
            if (ImGui::Button("Play Again [R]", ImVec2(284.f, 40.f))) {
                ballPos = glm::vec3{0.f, 22.f, 0.f};
                ballVel = glm::vec3{0.f};
                ballRot = glm::quat{1.f, 0.f, 0.f, 0.f};
                hasWon = false;
                hasLost = false;
                lostByFalling = false;
                timeRemaining = 20.0f;
                timerStarted = false;
            }
            ImGui::End();
        }

        ImGui::Render();

        updateUniformBuffer();

        device.resetFences(*inFlightFences[frameIndex]);

        commandBuffers[frameIndex].reset();
        recordCommandBuffer(commandBuffers[frameIndex], imageIndex, frameIndex);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[frameIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };
        queue.submit(submitInfo, *inFlightFences[frameIndex]);

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex
        };
        result = queue.presentKHR(presentInfo);
        if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else {
            assert(result == vk::Result::eSuccess);
        }

        frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void simulateBallPhysics(const float deltaTime, const glm::vec3 force = glm::vec3{0.f}) {
        // 1. Realistic Gravity & High Mass
        constexpr float gravity = -9.81f; // Standard Earth gravity
        constexpr float mass = 6.0f; // High mass makes the ball resist WASD inputs

        // Newton's Second Law: Acceleration = Force / Mass
        // Gravity is applied after because it affects all masses equally
        const glm::vec3 acc = (force / mass) + glm::vec3{0.f, gravity, 0.f};

        // Update velocity
        ballVel += acc * deltaTime;

        // Perform collision detection and resolution with all platforms
        bool onGround = false;
        constexpr float ballRadius = 1.0f;

        for (const auto &ent: platformEntities) {
            const auto *t = ent->getComponent<TransformComponent>();
            const auto *p = ent->getComponent<PlatformComponent>();
            if (!t || !p) continue;

            const glm::vec3 rot = t->getRotation();
            const glm::vec3 pos = t->getPosition();
            const glm::vec3 size = t->getScale();

            // Construct the rotation matrix of the platform
            glm::quat qx = glm::angleAxis(rot.x, glm::vec3{1.0f, 0.0f, 0.0f});
            glm::quat qy = glm::angleAxis(rot.y, glm::vec3{0.0f, 1.0f, 0.0f});
            glm::quat qz = glm::angleAxis(rot.z, glm::vec3{0.0f, 0.0f, 1.0f});
            glm::quat q = qz * qy * qx;
            glm::mat3 R = glm::mat3_cast(q);

            // Transform ball position to local space
            glm::vec3 relativePos = ballPos - pos;
            glm::vec3 localPos = glm::transpose(R) * relativePos;

            // Clamp local position to box half-extents
            glm::vec3 halfSize = size * 0.5f;
            glm::vec3 closestLocal = glm::clamp(localPos, -halfSize, halfSize);

            // Distance in local space
            glm::vec3 diff = localPos - closestLocal;
            float dist = glm::length(diff);

            if (dist < ballRadius) {
                // Collision detected!
                glm::vec3 normalLocal;
                float penetration;

                if (dist > 0.0001f) {
                    normalLocal = diff / dist;
                    penetration = ballRadius - dist;
                } else {
                    // Center is inside the box, push to the closest face
                    float dx = halfSize.x - std::abs(localPos.x);
                    float dy = halfSize.y - std::abs(localPos.y);
                    float dz = halfSize.z - std::abs(localPos.z);

                    if (dx < dy && dx < dz) {
                        normalLocal = glm::vec3{localPos.x >= 0.f ? 1.f : -1.f, 0.f, 0.f};
                        penetration = ballRadius + dx;
                    } else if (dy < dx && dy < dz) {
                        normalLocal = glm::vec3{0.f, localPos.y >= 0.f ? 1.f : -1.f, 0.f};
                        penetration = ballRadius + dy;
                    } else {
                        normalLocal = glm::vec3{0.f, 0.f, localPos.z >= 0.f ? 1.f : -1.f};
                        penetration = ballRadius + dz;
                    }
                }

                // Convert normal to world space
                glm::vec3 normalWorld = R * normalLocal;

                // Push ball out of the platform
                ballPos += normalWorld * penetration;

                // Adjust velocity
                float vn = glm::dot(ballVel, normalWorld);
                if (vn < 0.f) {
                    constexpr float restitution = 0.05f; // Slight bounce to prevent getting stuck
                    ballVel -= normalWorld * (vn * (1.f + restitution));
                }

                // Apply platform-specific behavior
                if (p->isGoal()) {
                    hasWon = true;
                }

                // Ground check: if normal points mostly upwards, mark as grounded
                if (normalWorld.y > 0.5f) {
                    onGround = true;
                }
            }
        }

        // 2. Momentum Preservation
        if (onGround) {
            // Lowered friction: a heavy ball keeps rolling once it has momentum
            constexpr float friction = 0.35f;
            ballVel.x -= ballVel.x * friction * deltaTime;
            ballVel.z -= ballVel.z * friction * deltaTime;
        } else {
            constexpr float airResistance = 0.0f;
            ballVel -= ballVel * airResistance * deltaTime;
        }

        // Update position
        ballPos += ballVel * deltaTime;

        // Ball fell off — trigger loss instead of respawning
        if (ballPos.y < -15.f) {
            ballVel = glm::vec3{0.f}; // freeze the ball
            hasLost = true;
            lostByFalling = true;
        }

        // Update visual rotation based on velocity — always, including while airborne
        {
            const auto horizontalVel = glm::vec3{ballVel.x, 0.f, ballVel.z};
            const float speed = glm::length(horizontalVel);
            if (speed > 0.001f) {
                const glm::vec3 rollDirection = glm::normalize(horizontalVel);
                const glm::vec3 rotAxis = glm::normalize(glm::cross(glm::vec3{0.f, 1.f, 0.f}, rollDirection));
                const float rotAngle = (speed * deltaTime) / ballRadius;
                const glm::quat deltaRot = glm::angleAxis(rotAngle, rotAxis);
                ballRot = deltaRot * ballRot;
                ballRot = glm::normalize(ballRot);
            }
        }
    }

    void update(const float deltaTime) {
        // Toggle control mode with TAB (prevent multiple triggers per keypress by tracking state)
        static bool tabPressedLast = false;
        const bool tabPressed = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
        if (tabPressed && !tabPressedLast) {
            ballControlMode = !ballControlMode;
            // Reset cursor state
            firstMouse = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        tabPressedLast = tabPressed;

        // Countdown timer — starts on first player move, then runs in any mode
        if (timerStarted && !hasWon && !hasLost) {
            timeRemaining -= deltaTime;
            if (timeRemaining <= 0.f) {
                timeRemaining = 0.f;
                hasLost = true;
            }
        }

        if (!ballControlMode) {
            // --- FLY MODE INPUTS ---
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                firstMouse = true;
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }

            if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
                double xpos = 0.;
                double ypos = 0.;
                glfwGetCursorPos(window, &xpos, &ypos);

                if (firstMouse) {
                    lastX = xpos;
                    lastY = ypos;
                    firstMouse = false;
                }

                const auto xoffset = static_cast<float>(xpos - lastX);
                const auto yoffset = static_cast<float>(lastY - ypos);
                lastX = xpos;
                lastY = ypos;

                constexpr float mouseSensitivity = 0.1f;
                yaw += xoffset * mouseSensitivity;
                pitch += yoffset * mouseSensitivity;

                pitch = std::clamp(pitch, -89.f, 89.f);
            }

            glm::vec3 front{0.f};
            front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            front.y = std::sin(glm::radians(pitch));
            front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            cameraFront = glm::normalize(front);

            constexpr float cameraSpeed = 2.5f;
            const float velocity = cameraSpeed * deltaTime;

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                cameraPos += cameraFront * velocity;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                cameraPos -= cameraFront * velocity;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
                cameraPos += cameraUp * velocity;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                cameraPos -= cameraUp * velocity;

            // Simulate ball physics passively in fly mode (only while game is active)
            if (!hasWon && !hasLost)
                simulateBallPhysics(deltaTime);
        } else {
            // --- BALL CONTROL MODE ---
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                firstMouse = true;
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }

            if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
                double xpos = 0.;
                double ypos = 0.;
                glfwGetCursorPos(window, &xpos, &ypos);

                if (firstMouse) {
                    lastX = xpos;
                    lastY = ypos;
                    firstMouse = false;
                }

                const auto xoffset = static_cast<float>(xpos - lastX);
                const auto yoffset = static_cast<float>(lastY - ypos);
                lastX = xpos;
                lastY = ypos;

                constexpr float mouseSensitivity = 0.1f;
                yaw += xoffset * mouseSensitivity;
                pitch += yoffset * mouseSensitivity;
                pitch = std::clamp(pitch, -89.f, 89.f);
            }

            glm::vec3 front{0.f};
            front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            front.y = std::sin(glm::radians(pitch));
            front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            cameraFront = glm::normalize(front);

            // Control ball with WASD relative to camera rotation on the horizontal plane
            glm::vec3 ballForce{0.f};
            const glm::vec3 flatForward = glm::normalize(glm::vec3{cameraFront.x, 0.f, cameraFront.z});
            const glm::vec3 flatRight = glm::normalize(glm::cross(flatForward, glm::vec3{0.f, 1.f, 0.f}));

            constexpr float forceMagnitude = 20.f; // force applied to move the ball
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                ballForce += flatForward * forceMagnitude;
                timerStarted = true;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                ballForce -= flatForward * forceMagnitude;
                timerStarted = true;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                ballForce -= flatRight * forceMagnitude;
                timerStarted = true;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                ballForce += flatRight * forceMagnitude;
                timerStarted = true;
            }


            // Only simulate physics while game is still active
            if (!hasLost && !hasWon)
                simulateBallPhysics(deltaTime, ballForce);

            // Third-person camera follow
            constexpr float followDistance = 6.f;
            cameraPos = ballPos - cameraFront * followDistance;
        }

        // Quick restart option
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            ballPos = glm::vec3{0.f, 22.f, 0.f};
            ballVel = glm::vec3{0.f};
            ballRot = glm::quat{1.f, 0.f, 0.f, 0.f};
            hasWon = false;
            hasLost = false;
            lostByFalling = false;
            timeRemaining = 20.0f;
            timerStarted = false;
        }

        // Sync and update ECS entities
        if (cameraEntity) {
            if (auto *cameraTransform = cameraEntity->getComponent<TransformComponent>()) {
                cameraTransform->setPosition(cameraPos);
                cameraTransform->setRotation(glm::vec3{glm::radians(pitch), -glm::radians(yaw + 90.f), 0.f});
            }
            cameraEntity->update(deltaTime);
        }

        if (ballEntity) {
            if (auto *ballTransform = ballEntity->getComponent<TransformComponent>()) {
                ballTransform->setPosition(ballPos);
                ballTransform->setRotation(glm::eulerAngles(ballRot));
            }
            ballEntity->update(deltaTime);
        }

        // Platform entities are static — update handles any future component logic
        for (const auto &ent: platformEntities)
            ent->update(deltaTime);
    }

    void updateUniformBuffer() const {
        const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);

        glm::mat4 proj;
        glm::mat4 view;
        if (cameraEntity) {
            if (auto *cameraComp = cameraEntity->getComponent<CameraComponent>()) {
                cameraComp->setAspectRatio(aspect);
                proj = cameraComp->getProjectionMatrix();
                view = cameraComp->getViewMatrix();
            } else {
                proj = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
                view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            }
        } else {
            proj = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        }

        constexpr auto model = glm::mat4{1.f};

        CameraUBO ubo{};
        ubo.model = model;
        ubo.view = view;
        ubo.proj = proj;
        ubo.lightDir = glm::vec4{1.f, 1.f, 1.f, 0.f};
        ubo.baseColor = glm::vec4{0.8f, 0.2f, 0.2f, 1.f};

        *static_cast<CameraUBO *>(uniformBuffersMapped[frameIndex]) = ubo;
    }

    void mainLoop() {
        auto lastTime = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            const auto currentTime = std::chrono::high_resolution_clock::now();
            const float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;

            update(deltaTime);
            drawFrame();
        }
        device.waitIdle();
    }

    static void transition_image_layout(
        const vk::raii::CommandBuffer &commandBuffer,
        const vk::Image image,
        const vk::ImageLayout old_layout,
        const vk::ImageLayout new_layout,
        const vk::AccessFlags2 src_access_mask,
        const vk::AccessFlags2 dst_access_mask,
        const vk::PipelineStageFlags2 src_stage_mask,
        const vk::PipelineStageFlags2 dst_stage_mask,
        const vk::ImageAspectFlags image_aspect_flags,
        const std::uint32_t level_count = 1) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = image_aspect_flags,
                .baseMipLevel = 0,
                .levelCount = level_count,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        const vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffer.pipelineBarrier2(dependency_info);
    }

    [[nodiscard]] std::uint32_t
    findMemoryType(const std::uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
        const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error{"failed to find suitable memory type!"};
    }

    void createImage(const std::uint32_t width, const std::uint32_t height, const std::uint32_t mipLevels,
                     const vk::Format format,
                     const vk::ImageTiling tiling,
                     const vk::ImageUsageFlags usage, const vk::MemoryPropertyFlags properties, vk::raii::Image &image,
                     vk::raii::DeviceMemory &imageMemory) const {
        const vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = mipLevels,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };
        image = vk::raii::Image{device, imageInfo};

        const vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        imageMemory = vk::raii::DeviceMemory{device, allocInfo};
        image.bindMemory(*imageMemory, 0);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(const vk::raii::Image &image, const vk::Format format,
                                                      const vk::ImageAspectFlags aspectFlags,
                                                      const std::uint32_t mipLevels = 1) const {
        const vk::ImageViewCreateInfo viewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {aspectFlags, 0, mipLevels, 0, 1}
        };
        return vk::raii::ImageView{device, viewInfo};
    }

    [[nodiscard]] std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
        const vk::DeviceSize size, const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties) const {
        const vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };
        vk::raii::Buffer buffer{device, bufferInfo};
        const vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        vk::raii::DeviceMemory bufferMemory{device, allocInfo};
        buffer.bindMemory(*bufferMemory, 0);
        return {std::move(buffer), std::move(bufferMemory)};
    }

    void copyBuffer(const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer,
                    const vk::DeviceSize size) const {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
        };
        const vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
        commandCopyBuffer.end();
        queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
        queue.waitIdle();
    }

    static std::uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
        assert(!availableFormats.empty());
        const auto formatIt = std::ranges::find_if(
            availableFormats,
            [](const auto &format) {
                return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace ==
                       vk::ColorSpaceKHR::eSrgbNonlinear;
            });
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes) {
        assert(
            std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::
                eFifo; }));
        return std::ranges::any_of(availablePresentModes,
                                   [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; })
                   ? vk::PresentModeKHR::eMailbox
                   : vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        return {
            std::clamp<std::uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<std::uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    static std::vector<const char *> getRequiredInstanceExtensions() {
        std::vector<const char *> extensions;

        std::uint32_t glfwExtensionCount = 0;
        const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);

        return extensions;
    }

    static GeometryData createSphereGeometry(const float radius, const std::uint32_t rings,
                                             const std::uint32_t sectors) {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;

        constexpr float pi = std::numbers::pi_v<float>;

        for (std::uint32_t r = 0; r <= rings; ++r) {
            const float theta = static_cast<float>(r) * pi / static_cast<float>(rings);
            const float sinTheta = std::sin(theta);
            const float cosTheta = std::cos(theta);

            for (std::uint32_t s = 0; s <= sectors; ++s) {
                const float phi = static_cast<float>(s) * 2.f * pi / static_cast<float>(sectors);
                const float x = std::cos(phi) * sinTheta;
                const float y = cosTheta;
                const float z = std::sin(phi) * sinTheta;

                const glm::vec3 position = glm::vec3{x * radius, y * radius, z * radius};
                const glm::vec3 normal = glm::vec3{x, y, z};
                const glm::vec2 texCoord = glm::vec2(static_cast<float>(s) / static_cast<float>(sectors),
                                                     static_cast<float>(r) / static_cast<float>(rings));

                // Normalize tangent vector around sphere Y axis
                glm::vec3 tangent = {-std::sin(phi), 0.f, std::cos(phi)};
                if (glm::length(tangent) > 0.0001f) {
                    tangent = glm::normalize(tangent);
                } else {
                    tangent = glm::vec3{1.f, 0.f, 0.f};
                }

                vertices.push_back(Vertex{
                    .position = position,
                    .normal = normal,
                    .texCoord = texCoord,
                    .tangent = glm::vec4{tangent, 1.0f}
                });
            }
        }

        for (std::uint32_t r = 0; r < rings; ++r) {
            for (std::uint32_t s = 0; s < sectors; ++s) {
                const std::uint32_t first = r * (sectors + 1) + s;
                const std::uint32_t second = first + sectors + 1;

                indices.push_back(first);
                indices.push_back(first + 1);
                indices.push_back(second);

                indices.push_back(second);
                indices.push_back(first + 1);
                indices.push_back(second + 1);
            }
        }

        return {.vertices = std::move(vertices), .indices = std::move(indices)};
    }

    static GeometryData createBoxGeometry() {
        GeometryData geom;

        constexpr std::array normals = {
            glm::vec3{0.f, 0.f, 1.f}, // Front (+Z)
            glm::vec3{0.f, 0.f, -1.f}, // Back (-Z)
            glm::vec3{1.f, 0.f, 0.f}, // Right (+X)
            glm::vec3{-1.f, 0.f, 0.f}, // Left (-X)
            glm::vec3{0.f, 1.f, 0.f}, // Top (+Y)
            glm::vec3{0.f, -1.f, 0.f} // Bottom (-Y)
        };

        constexpr std::array tangents = {
            glm::vec4{1.f, 0.f, 0.f, 1.f}, // Front
            glm::vec4{-1.f, 0.f, 0.f, 1.f}, // Back
            glm::vec4{0.f, 0.f, -1.f, 1.f}, // Right
            glm::vec4{0.f, 0.f, 1.f, 1.f}, // Left
            glm::vec4{1.f, 0.f, 0.f, 1.f}, // Top
            glm::vec4{1.f, 0.f, 0.f, 1.f} // Bottom
        };

        constexpr std::array positions = {
            glm::vec3{-0.5f, -0.5f, 0.5f},
            glm::vec3{0.5f, -0.5f, 0.5f},
            glm::vec3{0.5f, 0.5f, 0.5f},
            glm::vec3{-0.5f, 0.5f, 0.5f}
        };

        constexpr std::array uvs = {
            glm::vec2{0.f, 0.f},
            glm::vec2{1.f, 0.f},
            glm::vec2{1.f, 1.f},
            glm::vec2{0.f, 1.f}
        };

        // Front Face (+Z)
        for (int i = 0; i < 4; ++i) {
            geom.vertices.push_back({positions[i], normals[0], uvs[i], tangents[0]});
        }
        // Back Face (-Z)
        for (int i = 0; i < 4; ++i) {
            glm::vec3 p = positions[i];
            p.z = -p.z;
            p.x = -p.x;
            geom.vertices.push_back({p, normals[1], uvs[i], tangents[1]});
        }
        // Right Face (+X)
        for (int i = 0; i < 4; ++i) {
            glm::vec3 p = positions[i];
            float tmp = p.x;
            p.x = p.z;
            p.z = -tmp;
            geom.vertices.push_back({p, normals[2], uvs[i], tangents[2]});
        }
        // Left Face (-X)
        for (int i = 0; i < 4; ++i) {
            glm::vec3 p = positions[i];
            float tmp = p.x;
            p.x = -p.z;
            p.z = tmp;
            geom.vertices.push_back({p, normals[3], uvs[i], tangents[3]});
        }
        // Top Face (+Y)
        for (int i = 0; i < 4; ++i) {
            glm::vec3 p = positions[i];
            float tmp = p.y;
            p.y = p.z;
            p.z = -tmp;
            geom.vertices.push_back({p, normals[4], uvs[i], tangents[4]});
        }
        // Bottom Face (-Y)
        for (int i = 0; i < 4; ++i) {
            glm::vec3 p = positions[i];
            float tmp = p.y;
            p.y = -p.z;
            p.z = tmp;
            geom.vertices.push_back({p, normals[5], uvs[i], tangents[5]});
        }

        // Indices (Counter-Clockwise winding matching Vulkan pipeline frontFace)
        for (int f = 0; f < 6; ++f) {
            std::uint32_t base = f * 4;
            geom.indices.push_back(base + 0);
            geom.indices.push_back(base + 1);
            geom.indices.push_back(base + 2);
            geom.indices.push_back(base + 0);
            geom.indices.push_back(base + 2);
            geom.indices.push_back(base + 3);
        }

        return geom;
    }

    static std::vector<char> readFile(const std::string_view filename) {
        std::ifstream file{std::string{filename}, std::ios::ate | std::ios::binary};
        if (!file.is_open()) {
            throw std::runtime_error{"failed to open file: " + std::string{filename}};
        }

        const std::size_t fileSize = file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
        file.close();

        return buffer;
    }
};

int main() {
    try {
        MarbleGame app;
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
