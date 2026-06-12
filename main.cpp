#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_profiles.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include <limits>
#include <chrono>
#include <fstream>
#include <string_view>
#include <cstring>
#include <utility>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "common.hpp"

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;
constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;

    static constexpr vk::VertexInputBindingDescription getBindingDescription() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, pos)
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, normal)
            },
            vk::VertexInputAttributeDescription{
                .location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, color)
            },
            vk::VertexInputAttributeDescription{
                .location = 3,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, uv)
            }
        };
    }
};

struct GeometryData {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

class MarbleGame {
public:
    ~MarbleGame() {
        if (*device) {
            device.waitIdle();
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
    }

private:
    GLFWwindow *window = nullptr;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::SwapchainKHR swapChain = nullptr;

    std::vector<vk::Image> swapChainImages;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffers commandBuffers = nullptr;

    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    std::uint32_t currentFrame = 0;
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

    // --- Vulkan Resources for 3D Ball Rendering ---
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline linePipeline = nullptr;
    vk::raii::Buffer lineVertexBuffer = nullptr;
    vk::raii::DeviceMemory lineVertexBufferMemory = nullptr;

    vk::Format depthFormat = vk::Format::eUndefined;
    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::Buffer sphereVertexBuffer = nullptr;
    vk::raii::DeviceMemory sphereVertexBufferMemory = nullptr;
    std::uint32_t sphereIndexCount = 0;
    vk::raii::Buffer sphereIndexBuffer = nullptr;
    vk::raii::DeviceMemory sphereIndexBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSets descriptorSets = nullptr;

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
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandPool();
        createGeometryBuffers();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "MarbleGame",
            .applicationVersion = 1,
            .pEngineName = "Custom Engine",
            .engineVersion = 1,
            .apiVersion = VK_API_VERSION_1_3
        };

        std::uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        const std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        const vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        instance = vk::raii::Instance{context, createInfo};
    }

    void createSurface() {
        VkSurfaceKHR rawSurface = nullptr;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR{instance, rawSurface};
    }

    void pickPhysicalDevice() {
        constexpr VpProfileProperties profileProp{
            .profileName = VP_KHR_ROADMAP_2022_NAME,
            .specVersion = VP_KHR_ROADMAP_2022_SPEC_VERSION
        };

        const auto physicalDevices = instance.enumeratePhysicalDevices();
        for (const auto &physDevice: physicalDevices) {
            VkBool32 supported = VK_FALSE;
            if (vpGetPhysicalDeviceProfileSupport(*instance, *physDevice, &profileProp, &supported) == VK_SUCCESS &&
                supported) {
                physicalDevice = physDevice;
                return;
            }
        }

        throw std::runtime_error{"failed to find a physical device that supports the VP_KHR_roadmap_2022 profile!"};
    }

    void createLogicalDevice() {
        constexpr float queuePriority = 1.f;
        const vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = 0,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };

        const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        constexpr vk::PhysicalDeviceVulkan13Features vulkan13Features{
            .synchronization2 = vk::True,
            .dynamicRendering = vk::True
        };

        // Enable wide lines if supported by physical device
        const auto supportedFeatures = physicalDevice.getFeatures();
        vk::PhysicalDeviceFeatures enabledFeatures{};
        if (supportedFeatures.wideLines) {
            enabledFeatures.wideLines = vk::True;
        }

        const vk::DeviceCreateInfo createInfo{
            .pNext = &vulkan13Features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &enabledFeatures
        };

        device = vk::raii::Device{physicalDevice, createInfo};
        graphicsQueue = vk::raii::Queue{device, 0, 0};
    }

    void createSwapChain() {
        swapChainImageFormat = vk::Format::eB8G8R8A8Srgb;

        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        swapChainExtent = vk::Extent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};

        // Query surface capabilities to ensure valid image count
        const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);

        // Request minImageCount + 1 to enable triple buffering/smooth VSync,
        // which prevents the frame rate from dropping to 30 FPS if a VSync deadline is slightly missed.
        const std::uint32_t maxImages = (capabilities.maxImageCount > 0)
                                            ? capabilities.maxImageCount
                                            : std::numeric_limits<std::uint32_t>::max();
        const std::uint32_t imageCount = std::clamp(capabilities.minImageCount + 1, capabilities.minImageCount,
                                                    maxImages);

        // Use eFifo for standard vertical sync (capped at display refresh rate, e.g. 60 FPS)
        // support is guaranteed by the Vulkan specification.
        vk::PresentModeKHR selectedPresentMode = vk::PresentModeKHR::eFifo;

        const vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = imageCount,
            .imageFormat = swapChainImageFormat,
            .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = selectedPresentMode,
            .clipped = vk::True,
            .oldSwapchain = *swapChain
        };

        swapChain = vk::raii::SwapchainKHR{device, createInfo};
        swapChainImages = swapChain.getImages();
    }

    void createImageViews() {
        for (const auto &image: swapChainImages) {
            const vk::ImageViewCreateInfo createInfo{
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = swapChainImageFormat,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };
            swapChainImageViews.emplace_back(device, createInfo);
        }
    }

    void createDepthResources() {
        depthFormat = findDepthFormat();
        auto [image, memory] = createBufferImage(
            swapChainExtent.width,
            swapChainExtent.height,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        depthImage = std::move(image);
        depthImageMemory = std::move(memory);

        const vk::ImageViewCreateInfo viewInfo{
            .image = *depthImage,
            .viewType = vk::ImageViewType::e2D,
            .format = depthFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        depthImageView = vk::raii::ImageView{device, viewInfo};
    }

    vk::Format findDepthFormat() const {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, const vk::ImageTiling tiling,
                                   const vk::FormatFeatureFlags features) const {
        for (const auto &format: candidates) {
            const auto props = physicalDevice.getFormatProperties(format);

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
        constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
        };
        const vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding
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

        const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertModule, .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragModule, .pName = "main"
            }
        };

        const auto bindingDescription = Vertex::getBindingDescription();
        const auto attributeDescriptions = Vertex::getAttributeDescriptions();

        const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };

        constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False
        };

        constexpr vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.f
        };

        constexpr vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        constexpr vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        constexpr std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        const vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout
        };
        pipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

        const vk::PipelineRenderingCreateInfo renderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat
        };

        const vk::GraphicsPipelineCreateInfo pipelineInfo{
            .pNext = &renderingCreateInfo,
            .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicStateInfo,
            .layout = *pipelineLayout,
            .subpass = 0
        };

        graphicsPipeline = vk::raii::Pipeline{device, nullptr, pipelineInfo};

        // Create the line graphics pipeline
        vk::PipelineInputAssemblyStateCreateInfo lineInputAssembly{
            .topology = vk::PrimitiveTopology::eLineList,
            .primitiveRestartEnable = vk::False
        };

        // Disable face culling for lines and make them thicker
        vk::PipelineRasterizationStateCreateInfo lineRasterizer = rasterizer;
        lineRasterizer.cullMode = vk::CullModeFlagBits::eNone;
        lineRasterizer.lineWidth = 3.f;

        // Disable depth testing for debug lines so they draw on top of everything
        vk::PipelineDepthStencilStateCreateInfo lineDepthStencil = depthStencil;
        lineDepthStencil.depthTestEnable = vk::False;
        lineDepthStencil.depthWriteEnable = vk::False;

        vk::GraphicsPipelineCreateInfo linePipelineInfo = pipelineInfo;
        linePipelineInfo.pInputAssemblyState = &lineInputAssembly;
        linePipelineInfo.pRasterizationState = &lineRasterizer;
        linePipelineInfo.pDepthStencilState = &lineDepthStencil;

        linePipeline = vk::raii::Pipeline{device, nullptr, linePipelineInfo};
    }

    void createUniformBuffers() {
        const vk::DeviceSize bufferSize = sizeof(CameraUBO);

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            auto [buffer, memory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMapped.emplace_back(memory.mapMemory(0, bufferSize));
            uniformBuffersMemory.emplace_back(std::move(memory));
        }
    }

    void createDescriptorPool() {
        constexpr vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT
        };
        const vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize
        };
        descriptorPool = vk::raii::DescriptorPool{device, poolInfo};
    }

    void createDescriptorSets() {
        const std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
            .pSetLayouts = layouts.data()
        };
        descriptorSets = vk::raii::DescriptorSets{device, allocInfo};

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            const vk::DescriptorBufferInfo bufferInfo{
                .buffer = *uniformBuffers[i],
                .offset = 0,
                .range = sizeof(CameraUBO)
            };
            const vk::WriteDescriptorSet descriptorWrite{
                .dstSet = *descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
            };
            device.updateDescriptorSets(descriptorWrite, nullptr);
        }
    }

    void createCommandPool() {
        constexpr vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = 0
        };
        commandPool = vk::raii::CommandPool{device, poolInfo};
    }

    void createGeometryBuffers() {
        const GeometryData sphere = createSphereGeometry(1.f, 30, 30);
        sphereIndexCount = static_cast<std::uint32_t>(sphere.indices.size());

        // Vertex Buffer Allocation and Copy
        {
            const vk::DeviceSize bufferSize = sizeof(Vertex) * sphere.vertices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            void *data = stagingMemory.mapMemory(0, bufferSize);
            std::memcpy(data, sphere.vertices.data(), bufferSize);
            stagingMemory.unmapMemory();

            auto [vBuffer, vMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            sphereVertexBuffer = std::move(vBuffer);
            sphereVertexBufferMemory = std::move(vMemory);

            copyBuffer(stagingBuffer, sphereVertexBuffer, bufferSize);
        }

        // Index Buffer Allocation and Copy
        {
            const vk::DeviceSize bufferSize = sizeof(std::uint32_t) * sphere.indices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            void *data = stagingMemory.mapMemory(0, bufferSize);
            std::memcpy(data, sphere.indices.data(), bufferSize);
            stagingMemory.unmapMemory();

            auto [iBuffer, iMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            sphereIndexBuffer = std::move(iBuffer);
            sphereIndexBufferMemory = std::move(iMemory);

            copyBuffer(stagingBuffer, sphereIndexBuffer, bufferSize);
        }

        // Line Vertex Buffer (Static, Device-Local)
        {
            const float lineLength = 0.015f;
            const std::array<Vertex, 6> lineVertices = {
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

            const vk::DeviceSize bufferSize = sizeof(Vertex) * lineVertices.size();
            auto [stagingBuffer, stagingMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            void *data = stagingMemory.mapMemory(0, bufferSize);
            std::memcpy(data, lineVertices.data(), bufferSize);
            stagingMemory.unmapMemory();

            auto [vBuffer, vMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            );
            lineVertexBuffer = std::move(vBuffer);
            lineVertexBufferMemory = std::move(vMemory);

            copyBuffer(stagingBuffer, lineVertexBuffer, bufferSize);
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
        constexpr vk::SemaphoreCreateInfo semInfo{};
        constexpr vk::FenceCreateInfo fenceInfo{.flags = vk::FenceCreateFlagBits::eSignaled};

        // CPU-GPU synchronization and Image Acquisition (Indexed by currentFrame)
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            imageAvailableSemaphores.emplace_back(device, semInfo);
            inFlightFences.emplace_back(device, fenceInfo);
        }

        // Presentation synchronization (Indexed by imageIndex!)
        for (std::size_t i = 0; i < swapChainImages.size(); ++i) {
            renderFinishedSemaphores.emplace_back(device, semInfo);
        }
    }

    void recreateSwapChain() {
        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        swapChainImageViews.clear();
        createSwapChain();
        createImageViews();

        depthImageView = nullptr;
        depthImageMemory = nullptr;
        depthImage = nullptr;
        createDepthResources();

        renderFinishedSemaphores.clear();
        for (std::size_t i = 0; i < swapChainImages.size(); ++i) {
            constexpr vk::SemaphoreCreateInfo semInfo{};
            renderFinishedSemaphores.emplace_back(device, semInfo);
        }
    }

    void recordCommandBuffer(vk::raii::CommandBuffer &commandBuffer, const std::uint32_t imageIndex,
                             const std::uint32_t frameIdx) const {
        constexpr vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);

        const vk::ImageMemoryBarrier2 imageBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        const vk::ImageMemoryBarrier2 depthBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits2::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits2::eLateFragmentTests,
            .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .image = *depthImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        const std::array<vk::ImageMemoryBarrier2, 2> barriers = {imageBarrier, depthBarrier};
        const vk::DependencyInfo depInfo{
            .imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data()
        };
        commandBuffer.pipelineBarrier2(depInfo);

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

        commandBuffer.bindVertexBuffers(0, {*sphereVertexBuffer}, {0});
        commandBuffer.bindIndexBuffer(*sphereIndexBuffer, 0, vk::IndexType::eUint32);
        commandBuffer.drawIndexed(sphereIndexCount, 1, 0, 0, 0);

        // Draw debug lines at the center of the screen
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *linePipeline);
        commandBuffer.bindVertexBuffers(0, {*lineVertexBuffer}, {0});
        commandBuffer.draw(6, 1, 0, 0);

        commandBuffer.endRendering();

        const vk::ImageMemoryBarrier2 presentBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        const vk::DependencyInfo presentDepInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &presentBarrier
        };
        commandBuffer.pipelineBarrier2(presentDepInfo);

        commandBuffer.end();
    }

    void drawFrame() {
        [[maybe_unused]] const auto waitResult = device.waitForFences({*inFlightFences[currentFrame]}, VK_TRUE,
                                                                      std::numeric_limits<std::uint64_t>::max());

        device.resetFences({*inFlightFences[currentFrame]});

        const auto [acquireResult, imageIndex] = [&] {
            try {
                const auto [result, idx] = swapChain.acquireNextImage(
                    std::numeric_limits<std::uint64_t>::max(), *imageAvailableSemaphores[currentFrame], nullptr);
                return std::make_pair(result, idx);
            } catch ([[maybe_unused]] const vk::SystemError &err) {
                return std::make_pair(vk::Result::eErrorOutOfDateKHR, std::uint32_t{0});
            }
        }();

        if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error{"failed to acquire swap chain image!"};
        }

        updateUniformBuffer();

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex, currentFrame);

        constexpr std::array<vk::PipelineStageFlags, 1> waitStages = {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };

        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*imageAvailableSemaphores[currentFrame],
            .pWaitDstStageMask = waitStages.data(),
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };

        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex
        };

        try {
            const auto presentResult = graphicsQueue.presentKHR(presentInfo);
            if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
                framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            }
        } catch ([[maybe_unused]] const vk::SystemError &err) {
            framebufferResized = false;
            recreateSwapChain();
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void update(const float deltaTime) {
        // Handle cursor capture toggles
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            firstMouse = true;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }

        // Camera Look Controls (Mouse)
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            if (firstMouse) {
                lastX = xpos;
                lastY = ypos;
                firstMouse = false;
            }

            const float xoffset = static_cast<float>(xpos - lastX);
            const float yoffset = static_cast<float>(lastY - ypos);
            // Reversed since y-coordinates go from bottom to top
            lastX = xpos;
            lastY = ypos;

            constexpr float mouseSensitivity = 0.1f;
            yaw += xoffset * mouseSensitivity;
            pitch += yoffset * mouseSensitivity;

            // Constrain pitch rotation to prevent gimbal lock
            pitch = std::clamp(pitch, -89.f, 89.f);
        }

        // Update direction vector
        glm::vec3 front;
        front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        front.y = std::sin(glm::radians(pitch));
        front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);

        // Keyboard Controls for Movement: WASD & Space (Up) / Left Shift (Down)
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
    }

    void updateUniformBuffer() const {
        const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);

        // Perspective Projection Matrix: 45 degree vertical FOV, aspect ratio, near=0.1, far=10.0
        const glm::mat4 proj = glm::perspective(glm::radians(45.f), aspect, 0.1f, 10.f);

        // View Matrix: Uses dynamic camera position and look direction
        const glm::mat4 view = glm::lookAt(
            cameraPos,
            cameraPos + cameraFront,
            cameraUp
        );

        // Model Matrix: Identity (no rotation, ball stationary at 0, 0, 0)
        constexpr glm::mat4 model = glm::mat4(1.f);

        CameraUBO ubo{};
        ubo.model = model;
        ubo.view = view;
        ubo.proj = proj;
        ubo.lightDir = glm::vec4(1.f, 1.f, 1.f, 0.f); // directional light vector
        ubo.baseColor = glm::vec4(0.8f, 0.2f, 0.2f, 1.f); // red ball color

        std::memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
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

    // --- Vulkan Buffer Helpers ---
    [[nodiscard]] std::uint32_t findMemoryType(const std::uint32_t typeFilter,
                                               const vk::MemoryPropertyFlags properties) const {
        const auto memProps = physicalDevice.getMemoryProperties();
        for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error{"failed to find suitable memory type!"};
    }

    [[nodiscard]] std::pair<vk::raii::Image, vk::raii::DeviceMemory> createBufferImage(
        const std::uint32_t width,
        const std::uint32_t height,
        const vk::Format format,
        const vk::ImageTiling tiling,
        const vk::ImageUsageFlags usage,
        const vk::MemoryPropertyFlags properties
    ) const {
        const vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        vk::raii::Image image{device, imageInfo};
        const auto memRequirements = image.getMemoryRequirements();

        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory memory{device, allocInfo};
        image.bindMemory(*memory, 0);

        return {std::move(image), std::move(memory)};
    }

    [[nodiscard]] std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
        const vk::DeviceSize size,
        const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties
    ) const {
        const vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Buffer buffer{device, bufferInfo};
        const auto memRequirements = buffer.getMemoryRequirements();

        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory memory{device, allocInfo};
        buffer.bindMemory(*memory, 0);

        return {std::move(buffer), std::move(memory)};
    }

    void copyBuffer(const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer,
                    const vk::DeviceSize size) const {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffers tempCommandBuffers{device, allocInfo};
        const auto &tempCommandBuffer = tempCommandBuffers[0];

        constexpr vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        tempCommandBuffer.begin(beginInfo);

        const vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };
        tempCommandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

        tempCommandBuffer.end();

        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*tempCommandBuffer
        };

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    // --- Sphere Geometry Generator ---
    [[nodiscard]] static GeometryData createSphereGeometry(const float radius, const std::uint32_t rings,
                                                           const std::uint32_t sectors) {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;

        constexpr float pi = glm::pi<float>();

        // Generate positions and normals
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

        // Generate indices with a CW winding in local space,
        // which becomes CCW on screen after the negative viewport Y-flip.
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

    [[nodiscard]] static std::vector<char> readFile(const std::string_view filename) {
        std::vector<std::string> searchPaths = {
            std::string(filename),
            "cmake-build-debug/" + std::string(filename),
            "build/" + std::string(filename),
            "../" + std::string(filename),
            "../cmake-build-debug/" + std::string(filename),
            "../build/" + std::string(filename)
        };

        for (const auto &path: searchPaths) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (file.is_open()) {
                const std::size_t fileSize = static_cast<std::size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
                file.close();
                return buffer;
            }
        }

        throw std::runtime_error{"failed to open shader file: " + std::string(filename)};
    }
};

int main() {
    MarbleGame app;
    app.run();
    return 0;
}
