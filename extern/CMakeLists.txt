##################### BUILD EXTERNAL LIBRARIES #####################

# Build argparse
add_subdirectory(argparse)

# Build glad.
add_subdirectory(glad)

# Build glfw3.
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" FORCE)
add_subdirectory(glfw3)
add_library(glfw::glfw ALIAS glfw)

# Build glm.
add_subdirectory(glm)

# Build stb.
add_subdirectory(stb)