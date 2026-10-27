cmake_minimum_required(VERSION 3.20)
project(lffw_cubin_plugin LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "The cubin plugin supports Linux x86_64, including WSL2")
endif()
foreach(root IN ITEMS TENSORACLE_CPP_ROOT TENSORRT_ROOT LFFW_BUNDLE_ROOT)
    if(NOT DEFINED ${root} OR "${${root}}" STREQUAL "")
        message(FATAL_ERROR "Set ${root} to the verified build input")
    endif()
endforeach()
if(NOT CUDAToolkit_ROOT)
    find_path(CUDAToolkit_ROOT NAMES include/cuda.h
        HINTS ENV CUDAToolkit_ROOT
        PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES nvidia/cu13
        NO_DEFAULT_PATH REQUIRED)
endif()
include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleCubinPlugin.cmake")
find_package(Threads REQUIRED)
tensoracle_add_cubin_plugin(
    TARGET lffw_fc1_plugin
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${LFFW_BUNDLE_ROOT}"
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/fc1_registration.cpp" "${CMAKE_CURRENT_LIST_DIR}/fc1_plugin.cpp")
target_link_libraries(lffw_fc1_plugin PRIVATE Threads::Threads)
set_target_properties(lffw_fc1_plugin PROPERTIES SKIP_BUILD_RPATH TRUE)

include(CTest)
if(BUILD_TESTING)
    tensoracle_add_cubin_plugin(
        TARGET lffw_cubin_plugin
        TENSORRT_ROOT "${TENSORRT_ROOT}"
        BUNDLE_ROOT "${LFFW_BUNDLE_ROOT}"
        SOURCES "${CMAKE_CURRENT_LIST_DIR}/plugin.cpp" "${CMAKE_CURRENT_LIST_DIR}/qkv_plugin.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/fc1_plugin.cpp")
    target_link_libraries(lffw_cubin_plugin PRIVATE Threads::Threads)
    set_target_properties(lffw_cubin_plugin PROPERTIES SKIP_BUILD_RPATH TRUE)
    find_program(LFFW_NM nm REQUIRED)
    find_program(LFFW_READELF readelf REQUIRED)
    foreach(plugin IN ITEMS lffw_fc1_plugin lffw_cubin_plugin)
        add_test(NAME ${plugin}_registration_and_dependencies
            COMMAND "${CMAKE_COMMAND}"
                "-DPLUGIN_LIBRARY=$<TARGET_FILE:${plugin}>"
                "-DNM=${LFFW_NM}" "-DREADELF=${LFFW_READELF}"
                -P "${CMAKE_CURRENT_LIST_DIR}/tests/check_plugin.cmake")
        set_tests_properties(${plugin}_registration_and_dependencies PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-007;SP-LFFASTESTWHISPER-011")
    endforeach()

    # A separate test observer interposes only calls originating in the plugin.
    # It is never linked into the production library.
    add_library(lffw_qkv_driver_observer SHARED tests/qkv_driver_observer.cpp)
    target_compile_features(lffw_qkv_driver_observer PRIVATE cxx_std_17)
    target_link_libraries(lffw_qkv_driver_observer PRIVATE
        tensoracle_internal_cuda_driver Threads::Threads ${CMAKE_DL_LIBS})

    tensoracle_add_cubin_plugin(
        TARGET lffw_qkv_v1_fixture
        TENSORRT_ROOT "${TENSORRT_ROOT}"
        BUNDLE_ROOT "${LFFW_BUNDLE_ROOT}"
        SOURCES "${CMAKE_CURRENT_LIST_DIR}/tests/qkv_v1_fixture.cpp")
    target_link_libraries(lffw_qkv_v1_fixture PRIVATE Threads::Threads)
    set_target_properties(lffw_qkv_v1_fixture PROPERTIES SKIP_BUILD_RPATH TRUE)
endif()
