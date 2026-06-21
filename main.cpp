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

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;
constexpr std::int32_t MAX_FRAMES_IN_FLIGHT = 2;


struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)},
            vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)},
            vk::VertexInputAttributeDescription{3, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)}
        };
    }
};

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

    // --- Control and Physics State ---
    bool ballControlMode = false;
    glm::vec3 ballPos = glm::vec3(0.f, 2.f, 0.f);
    glm::vec3 ballVel = glm::vec3(0.f);
    glm::quat ballRot = glm::quat(1.f, 0.f, 0.f, 0.f);

    // --- Textures State ---
    vk::raii::Image ballTextureImage = nullptr;
    vk::raii::DeviceMemory ballTextureImageMemory = nullptr;
    vk::raii::ImageView ballTextureImageView = nullptr;

    vk::raii::Image floorTextureImage = nullptr;
    vk::raii::DeviceMemory floorTextureImageMemory = nullptr;
    vk::raii::ImageView floorTextureImageView = nullptr;

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
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

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
                .pStages = shaderStages,
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
                        .pStages = shaderStages,
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
                    .pos = glm::vec3(0.f), .normal = glm::vec3(0, 1, 0), .color = glm::vec3(1.f, 0.f, 0.f),
                    .uv = glm::vec2(0.f)
                },
                Vertex{
                    .pos = glm::vec3(lineLength, 0.f, 0.f), .normal = glm::vec3(0, 1, 0),
                    .color = glm::vec3(1.f, 0.f, 0.f), .uv = glm::vec2(0.f)
                },
                // Y-axis (Green)
                Vertex{
                    .pos = glm::vec3(0.f), .normal = glm::vec3(0, 0, 1), .color = glm::vec3(0.f, 1.f, 0.f),
                    .uv = glm::vec2(0.f)
                },
                Vertex{
                    .pos = glm::vec3(0.f, lineLength, 0.f), .normal = glm::vec3(0, 0, 1),
                    .color = glm::vec3(0.f, 1.f, 0.f), .uv = glm::vec2(0.f)
                },
                // Z-axis (Blue)
                Vertex{
                    .pos = glm::vec3(0.f), .normal = glm::vec3(1, 0, 0), .color = glm::vec3(0.f, 0.f, 1.f),
                    .uv = glm::vec2(0.f)
                },
                Vertex{
                    .pos = glm::vec3(0.f, 0.f, lineLength), .normal = glm::vec3(1, 0, 0),
                    .color = glm::vec3(0.f, 0.f, 1.f), .uv = glm::vec2(0.f)
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
                    .pos = glm::vec3(-planeSize, planeY, -planeSize), .normal = glm::vec3(0.f, 1.f, 0.f),
                    .color = glm::vec3(0.6f, 0.6f, 0.6f), .uv = glm::vec2(0.f, 0.f)
                },
                Vertex{
                    .pos = glm::vec3(planeSize, planeY, -planeSize), .normal = glm::vec3(0.f, 1.f, 0.f),
                    .color = glm::vec3(0.6f, 0.6f, 0.6f), .uv = glm::vec2(planeSize, 0.f)
                },
                Vertex{
                    .pos = glm::vec3(planeSize, planeY, planeSize), .normal = glm::vec3(0.f, 1.f, 0.f),
                    .color = glm::vec3(0.6f, 0.6f, 0.6f), .uv = glm::vec2(planeSize, planeSize)
                },
                Vertex{
                    .pos = glm::vec3(-planeSize, planeY, planeSize), .normal = glm::vec3(0.f, 1.f, 0.f),
                    .color = glm::vec3(0.6f, 0.6f, 0.6f), .uv = glm::vec2(0.f, planeSize)
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

            const std::array<vk::Offset3D, 2> srcOffsets = {
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{mipWidth, mipHeight, 1}
            };
            const std::array<vk::Offset3D, 2> dstOffsets = {
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
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * 2}
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
                .float32 = std::array<float, 4>{0.1f, 0.2f, 0.3f, 1.f}
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

        // Draw the plane
        {
            glm::mat4 planeModel = glm::mat4(1.0f);
            PushConstants pcs{
                .model = planeModel,
                .textureIndex = TextureIndex::Floor
            };
            commandBuffer.pushConstants<PushConstants>(*pipelineLayout,
                                                       vk::ShaderStageFlagBits::eVertex |
                                                       vk::ShaderStageFlagBits::eFragment, 0, pcs);
            commandBuffer.bindVertexBuffers(0, {*planeVertexBuffer}, {0});
            commandBuffer.bindIndexBuffer(*planeIndexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(planeIndexCount, 1, 0, 0, 0);
        }

        // Draw the sphere
        {
            glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), ballPos) * glm::mat4_cast(ballRot);
            PushConstants pcs{
                .model = sphereModel,
                .textureIndex = TextureIndex::Ball
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
            glm::mat4 identity = glm::mat4(1.0f);
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
            ImGui::Text("Ball Speed: %.2f m/s", glm::length(glm::vec3(ballVel.x, 0.f, ballVel.z)));
            ImGui::Text("Controls: WASD=Roll Ball, Mouse=Orbit Camera, Space=Jump");
        }
        ImGui::Text("Facing: %.1f / %.1f (Yaw / Pitch)", displayYaw, pitch);
        ImGui::End();

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

    void simulateBallPhysics(const float deltaTime, const glm::vec3 force = glm::vec3(0.f)) {
        // Gravity
        constexpr float gravity = -9.81f;
        const glm::vec3 acc = force + glm::vec3(0.f, gravity, 0.f);

        // Update velocity
        ballVel += acc * deltaTime;

        // Apply ground friction/damping on the ground
        constexpr float planeY = -1.f;
        constexpr float ballRadius = 1.f;
        const bool onGround = (ballPos.y <= planeY + ballRadius + 0.001f);

        if (onGround) {
            // Apply ground friction (drag) to horizontal movement
            constexpr float friction = 2.f;
            ballVel.x -= ballVel.x * friction * deltaTime;
            ballVel.z -= ballVel.z * friction * deltaTime;
        } else {
            // Air resistance
            constexpr float airResistance = 0.1f;
            ballVel -= ballVel * airResistance * deltaTime;
        }

        // Update position
        ballPos += ballVel * deltaTime;

        // Plane collision resolution
        if (ballPos.y < planeY + ballRadius) {
            ballPos.y = planeY + ballRadius;
            // If dropping down, bounce
            if (ballVel.y < 0.f) {
                constexpr float restitution = 0.3f; // bounce factor
                ballVel.y = -ballVel.y * restitution;
                if (std::abs(ballVel.y) < 0.2f) {
                    ballVel.y = 0.f;
                }
            }
        }

        // Keep ball within bounds of the plane
        constexpr float planeBoundary = 24.5f; // plane is 50x50, so bounds from -25 to 25
        if (ballPos.x < -planeBoundary) {
            ballPos.x = -planeBoundary;
            ballVel.x = -ballVel.x * 0.5f;
        }
        if (ballPos.x > planeBoundary) {
            ballPos.x = planeBoundary;
            ballVel.x = -ballVel.x * 0.5f;
        }
        if (ballPos.z < -planeBoundary) {
            ballPos.z = -planeBoundary;
            ballVel.z = -ballVel.z * 0.5f;
        }
        if (ballPos.z > planeBoundary) {
            ballPos.z = planeBoundary;
            ballVel.z = -ballVel.z * 0.5f;
        }

        // Update visual rotation based on rolling distance
        if (onGround) {
            const auto velocityOnGround = glm::vec3(ballVel.x, 0.f, ballVel.z);
            const float speed = glm::length(velocityOnGround);
            if (speed > 0.001f) {
                const glm::vec3 rollDirection = glm::normalize(velocityOnGround);
                // Rotation axis is perpendicular to roll direction and up vector (0, 1, 0)
                const glm::vec3 rotAxis = glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), rollDirection));
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

            // Simulate ball physics passively in fly mode
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
            const glm::vec3 flatForward = glm::normalize(glm::vec3(cameraFront.x, 0.f, cameraFront.z));
            const glm::vec3 flatRight = glm::normalize(glm::cross(flatForward, glm::vec3(0.f, 1.f, 0.f)));

            constexpr float forceMagnitude = 20.f; // force applied to move the ball
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                ballForce += flatForward * forceMagnitude;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                ballForce -= flatForward * forceMagnitude;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                ballForce -= flatRight * forceMagnitude;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                ballForce += flatRight * forceMagnitude;


            simulateBallPhysics(deltaTime, ballForce);

            // Third-person camera follow
            constexpr float followDistance = 6.f;
            cameraPos = ballPos - cameraFront * followDistance;
        }
    }

    void updateUniformBuffer() const {
        const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
        const glm::mat4 proj = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
        const glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        constexpr glm::mat4 model = glm::mat4{1.f};

        CameraUBO ubo{};
        ubo.model = model;
        ubo.view = view;
        ubo.proj = proj;
        ubo.lightDir = glm::vec4(1.f, 1.f, 1.f, 0.f);
        ubo.baseColor = glm::vec4(0.8f, 0.2f, 0.2f, 1.f);

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

                vertices.push_back(Vertex{
                    .pos = glm::vec3(x * radius, y * radius, z * radius),
                    .normal = glm::vec3(x, y, z),
                    .color = glm::vec3(1.f, 1.f, 1.f),
                    .uv = glm::vec2(static_cast<float>(s) / static_cast<float>(sectors),
                                    static_cast<float>(r) / static_cast<float>(rings))
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
