cmake_minimum_required(VERSION 3.10)

project(GLSL-CS-PathTracer VERSION 0.0.1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

set(CSPT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/source")
set(CSPT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/source")
set(CSPT_EXTERN_DIR "${PROJECT_SOURCE_DIR}/extern")
set(CSPT_SHADER_DIR "${PROJECT_SOURCE_DIR}/shader")
set(CSPT_SCENE_DIR "${PROJECT_SOURCE_DIR}/scene")
set(CSPT_SOURCES
    "${CSPT_SOURCE_DIR}/glslcs_pt.cpp"
)

# Build external libraries.
add_subdirectory(extern)

# Build main target.
add_executable(glslcs-pt ${CSPT_SOURCES})
target_include_directories(glslcs-pt PRIVATE ${CSPT_INCLUDE_DIR})
target_link_libraries(glslcs-pt argparse::argparse glad::glad glfw::glfw glm::glm stb::stb)

# Turn off volatile warning for GLM.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(glslcs-pt PRIVATE "-Wno-volatile")
endif()

set(CSPT_SHADER_OUTPUT_DIR "${PROJECT_BINARY_DIR}/shader")
set(CSPT_SCENE_OUTPUT_DIR "${PROJECT_BINARY_DIR}/scene")

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CSPT_SHADER_OUTPUT_DIR "${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/shader")
    set(CSPT_SCENE_OUTPUT_DIR "${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/scene")
endif()

# Copy data after build.
add_custom_target(copy_data)
add_custom_command(TARGET copy_data POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory "${CSPT_SHADER_DIR}" "${CSPT_SHADER_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory  "${CSPT_SCENE_DIR}" "${CSPT_SCENE_OUTPUT_DIR}"
)
add_dependencies(glslcs-pt copy_data)