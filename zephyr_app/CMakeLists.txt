# ZephyrMLForge TFLite Micro Application
# Based on TensorFlow Lite Micro hello_world sample structure
#
# The pipeline copies this tree per run and overwrites src/model.cpp, src/model.hpp
# and src/app_config.h. Nothing else differs between a pipeline build and a manual
# one, so a failure here reproduces with a plain west build.

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(zephyr_ml_forge)

# TFLite Micro uses function-local statics that Zephyr cannot guard.
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# Structure mirrors the TFLite Micro hello_world sample:
#   - main.c: Minimal entry point
#   - main_functions.cpp: Interpreter setup, benchmark and host request handling
#   - model.cpp: Compiled model data (generated from .tflite)
#   - output_handler.cpp: Metrics output for pipeline parsing
target_sources(app PRIVATE
    src/main.c
    src/main_functions.cpp
    src/model.cpp
    src/output_handler.cpp
)

target_include_directories(app PRIVATE
    src
)
