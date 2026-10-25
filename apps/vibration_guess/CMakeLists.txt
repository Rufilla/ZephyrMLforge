cmake_minimum_required(VERSION 3.20.0)

#  The click's pins, sensor & drivers come from the pipeline firmware, so
# both read the accelerometer the same way.
set(ZEPHYR_APP_DIR ${CMAKE_CURRENT_LIST_DIR}/../../zephyr_app)
list(APPEND EXTRA_DTC_OVERLAY_FILE ${ZEPHYR_APP_DIR}/vibration.overlay)
list(APPEND EXTRA_CONF_FILE
    ${CMAKE_CURRENT_LIST_DIR}/../common/tflm.conf
    ${ZEPHYR_APP_DIR}/vibration.conf
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(vibration_guess)

include(${CMAKE_CURRENT_LIST_DIR}/../common/app.cmake)

target_sources(app PRIVATE
    src/main.cpp
    src/model.cpp
    ${ZEPHYR_APP_DIR}/src/vibration.cpp
)
target_include_directories(app PRIVATE src ${ZEPHYR_APP_DIR}/src)

#  Must match the recording the model was trained on: 128 samples at 1 kHz.
target_compile_definitions(app PRIVATE
    MLFORGE_VIBRATION=1
    MLFORGE_WINDOW_SAMPLES=128
    MLFORGE_SAMPLE_RATE_HZ=1000
)
