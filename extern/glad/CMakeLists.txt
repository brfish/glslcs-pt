cmake_minimum_required(VERSION 3.8)

project(glad)

set(CMAKE_C_STANDARD 99)

add_library(glad STATIC "src/glad.c")
add_library(glad::glad ALIAS glad)
target_include_directories(glad PUBLIC "include")