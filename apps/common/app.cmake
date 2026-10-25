# TFLite Micro uses function-local statics that Zephyr cannot guard.
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

target_sources(app PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inference.cpp)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_LIST_DIR})
