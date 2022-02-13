##################### BUILD EXTERNAL LIBRARIES #####################

# Build argparse
add_subdirectory(argparse)

# Build glad.
add_library(glad STATIC "glad/src/glad.c")
add_library(glad::glad ALIAS glad)
target_include_directories(glad PUBLIC "glad/include")

# Build glfw3.
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" FORCE)
add_subdirectory(glfw3)
add_library(glfw::glfw ALIAS glfw)

# Build glm.
add_subdirectory(glm)

# Build stb.
add_library(stb_image STATIC "stb/src/stb_image.c")
add_library(stb::image ALIAS stb_image)
target_include_directories(stb_image PUBLIC "stb/include")
add_library(stb_image_write STATIC "stb/src/stb_image_write.c")
add_library(stb::image_write ALIAS stb_image_write)
target_include_directories(stb_image_write PUBLIC "stb/include")
add_library(stb INTERFACE)
add_library(stb::stb ALIAS stb)
target_link_libraries(stb INTERFACE stb_image stb_image_write)