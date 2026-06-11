#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_profiles.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include <limits>

constexpr std::uint32_t WIDTH = 800;
constexpr std::uint32_t HEIGHT = 600;
constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

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
    }

    void initVulkan() {
        createInstance();
        createSurface();
        const auto physDevice = pickPhysicalDevice();
        createLogicalDevice(physDevice);
        createSwapChain();
        createImageViews();
        createCommandPool();
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

    vk::raii::PhysicalDevice pickPhysicalDevice() const {
        constexpr VpProfileProperties profileProp{
            .profileName = VP_KHR_ROADMAP_2022_NAME,
            .specVersion = VP_KHR_ROADMAP_2022_SPEC_VERSION
        };

        const auto physicalDevices = instance.enumeratePhysicalDevices();
        for (const auto &physDevice: physicalDevices) {
            VkBool32 supported = VK_FALSE;
            if (vpGetPhysicalDeviceProfileSupport(*instance, *physDevice, &profileProp, &supported) == VK_SUCCESS &&
                supported) {
                return physDevice;
            }
        }

        throw std::runtime_error{"failed to find a physical device that supports the VP_KHR_roadmap_2022 profile!"};
    }

    void createLogicalDevice(const vk::raii::PhysicalDevice &physDevice) {
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

        const vk::DeviceCreateInfo createInfo{
            .pNext = &vulkan13Features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data()
        };

        device = vk::raii::Device{physDevice, createInfo};
        graphicsQueue = vk::raii::Queue{device, 0, 0};
    }

    void createSwapChain() {
        swapChainImageFormat = vk::Format::eB8G8R8A8Srgb;

        std::int32_t width = 0;
        std::int32_t height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        swapChainExtent = vk::Extent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};

        const vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = 3,
            .imageFormat = swapChainImageFormat,
            .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = vk::PresentModeKHR::eFifo,
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

    void createCommandPool() {
        constexpr vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = 0
        };
        commandPool = vk::raii::CommandPool{device, poolInfo};
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

        renderFinishedSemaphores.clear();
        for (std::size_t i = 0; i < swapChainImages.size(); ++i) {
            constexpr vk::SemaphoreCreateInfo semInfo{};
            renderFinishedSemaphores.emplace_back(device, semInfo);
        }
    }

    void recordCommandBuffer(vk::raii::CommandBuffer &commandBuffer, const std::uint32_t imageIndex) const {
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

        const vk::DependencyInfo depInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &imageBarrier
        };
        commandBuffer.pipelineBarrier2(depInfo);

        // FIX: Explicitly initialize the union using designated initialization
        constexpr vk::ClearValue clearValue{
            .color = vk::ClearColorValue{
                .float32 = std::array<float, 4>{0.1f, 0.2f, 0.3f, 1.f}
            }
        };

        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eNone,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearValue
        };

        const vk::RenderingInfo renderingInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment
        };

        commandBuffer.beginRendering(renderingInfo);

        // ---> Draw calls go here <---

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

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

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
            // FIX: Index the signal semaphore by imageIndex, not currentFrame
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };

        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            // FIX: Wait on the semaphore tied to this specific image
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

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        device.waitIdle();
    }
};

int main() {
    MarbleGame app;
    app.run();
    return 0;
}
