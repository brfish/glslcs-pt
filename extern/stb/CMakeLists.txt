cmake_minimum_required(VERSION 3.8)

project(stb)

set(CMAKE_C_STANDARD 99)

add_library(stb_image STATIC "src/stb_image.c")
add_library(stb::image ALIAS stb_image)
target_include_directories(stb_image PUBLIC "include")

add_library(stb_image_write STATIC "src/stb_image_write.c")
add_library(stb::image_write ALIAS stb_image_write)
target_include_directories(stb_image_write PUBLIC "include")

add_library(stb INTERFACE)
add_library(stb::stb ALIAS stb)
target_link_libraries(stb INTERFACE stb_image stb_image_write)