cmake_minimum_required(VERSION 3.20.0)

list(APPEND EXTRA_CONF_FILE ${CMAKE_CURRENT_LIST_DIR}/../common/tflm.conf)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sine_predict)

include(${CMAKE_CURRENT_LIST_DIR}/../common/app.cmake)

target_sources(app PRIVATE src/main.cpp src/model.cpp)
target_include_directories(app PRIVATE src)
