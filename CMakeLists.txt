cmake_minimum_required(VERSION 3.28)
project(MarbleGame)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# 1. Find Vulkan SDK
find_package(Vulkan REQUIRED)

# 2. Automatically download and build GLFW
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# Fetch GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch ImGui
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG docking
        SOURCE_SUBDIR "" 
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(imgui PUBLIC glfw)

# 3. Create the executable
add_executable(MarbleGame main.cpp)
target_include_directories(MarbleGame PRIVATE "C:/vcpkg/packages/stb_x64-windows/include")

# 4. Link the necessary libraries
target_link_libraries(MarbleGame PRIVATE
        Vulkan::Vulkan
        glfw
        imgui
        glm::glm
)

# 5. Enable dynamic dispatch for Vulkan-Hpp
target_compile_definitions(MarbleGame PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_CONSTRUCTORS
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
)

# 6. Slang compilation
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(MAIN_SLANG "${SHADER_DIR}/main.slang")
set(COMMON_SLANG "${SHADER_DIR}/common.slang")
set(VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/main_vert.spv")
set(FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/main_frag.spv")

add_custom_command(
        OUTPUT ${VERT_SPV}
        COMMAND ${SLANGC_EXECUTABLE} ${MAIN_SLANG} -target spirv -entry vertexMain -stage vertex -o ${VERT_SPV}
        DEPENDS ${MAIN_SLANG} ${COMMON_SLANG}
)
add_custom_command(
        OUTPUT ${FRAG_SPV}
        COMMAND ${SLANGC_EXECUTABLE} ${MAIN_SLANG} -target spirv -entry fragmentMain -stage fragment -o ${FRAG_SPV}
        DEPENDS ${MAIN_SLANG} ${COMMON_SLANG}
)

add_custom_target(CompileAllShaders DEPENDS ${VERT_SPV} ${FRAG_SPV})
add_dependencies(MarbleGame CompileAllShaders)

# 7. Copy assets and shaders to the executable directory
add_custom_command(TARGET MarbleGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/textures"
        "$<TARGET_FILE_DIR:MarbleGame>/textures"
)

add_custom_command(TARGET MarbleGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${VERT_SPV}"
        "${FRAG_SPV}"
        "$<TARGET_FILE_DIR:MarbleGame>/"
)
