cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/conan/bootstrap.cmake")
project(lffw_whisper LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "The cubin plugin supports Linux x86_64, including WSL2")
endif()
# The model library can be built independently of cubin production and Python.
option(LFFW_BUILD_NETWORK "Build the tensoracle v2 Whisper model library" OFF)
option(LFFW_BUILD_PLUGINS "Build cubin plugin libraries" ON)
option(LFFW_BUILD_RUNTIME "Build the Whisper inference C ABI" ON)
option(LFFW_RUNTIME_DIAGNOSTICS "Build development-only inference diagnostics" OFF)
if(LFFW_RUNTIME_DIAGNOSTICS AND NOT LFFW_BUILD_RUNTIME)
    message(FATAL_ERROR "LFFW_RUNTIME_DIAGNOSTICS requires LFFW_BUILD_RUNTIME")
endif()
if(LFFW_BUILD_RUNTIME)
    set(LFFW_BUILD_NETWORK ON)
endif()
# Both SDKs come from the single Conan installation and generated CMake packages.
find_package(tensoracle 2.6.0 CONFIG REQUIRED)
find_package(TensorRT CONFIG REQUIRED)
find_package(Threads REQUIRED)
include(CTest)
set(LFFW_TEST_TENSORRT_LIBRARY_DIR "${LFFW_TENSORRT_ROOT}/lib" CACHE PATH
    "TensorRT runtime libraries for native graph integration tests")
if(BUILD_TESTING)
    find_program(LFFW_CUDA_READELF readelf REQUIRED)
    function(lffw_test_static_cuda_runtime target)
        set(link_map "${CMAKE_CURRENT_BINARY_DIR}/${target}.link.map")
        target_link_options(${target} PRIVATE "LINKER:-Map=${link_map}")
        add_test(NAME ${target}_static_cuda_runtime
            COMMAND "${CMAKE_COMMAND}" "-DLIBRARY=$<TARGET_FILE:${target}>"
                "-DLINK_MAP=${link_map}" "-DTOOLKIT_ROOT=${CUDAToolkit_ROOT}"
                "-DREADELF=${LFFW_CUDA_READELF}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_static_cuda_runtime.cmake")
        set_tests_properties(${target}_static_cuda_runtime PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-055")
    endfunction()
endif()
if(LFFW_BUILD_NETWORK OR LFFW_BUILD_PLUGINS)
    find_package(CUDAToolkit 13.2 REQUIRED)
endif()
if(LFFW_BUILD_NETWORK)
    set(LFFW_NETWORK_TRT_INCLUDE "${LFFW_TENSORRT_ROOT}/include")
    find_path(LFFW_NETWORK_CUDA_INCLUDE cuda_runtime_api.h
        PATHS "${CUDAToolkit_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    tensoracle_find_cuda(OUT_INCLUDE_DIR lffw_gpu_cuda_include
        OUT_DRIVER_LIBRARY lffw_gpu_cuda_driver)
    add_library(lffw_network STATIC src/whisper_network.cpp src/network_graph.cpp)
    target_compile_features(lffw_network PUBLIC cxx_std_17)
    target_include_directories(lffw_network PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(lffw_network SYSTEM PRIVATE "${LFFW_NETWORK_TRT_INCLUDE}" "${LFFW_NETWORK_CUDA_INCLUDE}")
    target_link_libraries(lffw_network PRIVATE tensoracle::tensoracle TensorRT::nvinfer
         tensoracle_cuda_driver ${CMAKE_DL_LIBS})
    set_target_properties(lffw_network PROPERTIES POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    if(BUILD_TESTING)
        add_executable(lffw_network_fc1_contract tests/fc1_graph_contract.cpp)
        target_compile_features(lffw_network_fc1_contract PRIVATE cxx_std_17)
        add_test(NAME lffw_network_fc1_contract COMMAND lffw_network_fc1_contract)
        set_tests_properties(lffw_network_fc1_contract PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-002;SP-LFFASTESTWHISPER-004")
        add_executable(lffw_network_gpu tests/network_gpu.cpp)
        target_link_libraries(lffw_network_gpu PRIVATE lffw_network tensoracle::tensoracle TensorRT::nvinfer tensoracle_cuda_driver)
        target_include_directories(lffw_network_gpu PRIVATE src)
        target_include_directories(lffw_network_gpu SYSTEM PRIVATE
            "${LFFW_NETWORK_TRT_INCLUDE}" "${LFFW_NETWORK_CUDA_INCLUDE}")
        foreach(graph IN ITEMS mel encoder decoder sampled)
            foreach(precision IN ITEMS fp32 fp16 fp8)
                if(graph STREQUAL "sampled")
                    set(lffw_test_history 448)
                else()
                    set(lffw_test_history 1)
                endif()
                if(precision STREQUAL "fp32" OR graph STREQUAL "mel")
                    set(lffw_test_kind values)
                    set(lffw_test_requirements "SP-LFFASTESTWHISPER-002;SP-LFFASTESTWHISPER-004;SP-LFFASTESTWHISPER-008")
                else()
                    set(lffw_test_kind structure)
                    set(lffw_test_requirements "SP-LFFASTESTWHISPER-004;SP-LFFASTESTWHISPER-008;SP-LFFASTESTWHISPER-016")
                endif()
                add_test(NAME lffw_network_${graph}_${precision}_${lffw_test_kind}_gpu
                    COMMAND lffw_network_gpu ${graph} ${precision} 5 ${lffw_test_history})
                set_tests_properties(lffw_network_${graph}_${precision}_${lffw_test_kind}_gpu PROPERTIES
                    SKIP_RETURN_CODE 77 TIMEOUT 5400 LABELS "gpu;${lffw_test_requirements}")
            endforeach()
        endforeach()
        # Consume the model target without backend/CUDA compile inputs.
        add_executable(lffw_network_public_header tests/network_public_header.cpp)
        target_link_libraries(lffw_network_public_header PRIVATE lffw_network)
        add_test(NAME lffw_network_public_header COMMAND lffw_network_public_header)
        set_tests_properties(lffw_network_public_header PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-031;SP-LFFASTESTWHISPER-033")
        add_executable(lffw_network_contract tests/network_contract.cpp)
        target_include_directories(lffw_network_contract PRIVATE src)
        target_include_directories(lffw_network_contract SYSTEM PRIVATE "${LFFW_NETWORK_TRT_INCLUDE}" "${LFFW_NETWORK_CUDA_INCLUDE}")
        target_link_libraries(lffw_network_contract PRIVATE lffw_network tensoracle::tensoracle TensorRT::nvinfer)
        add_test(NAME lffw_network_contract COMMAND lffw_network_contract
            "${CMAKE_CURRENT_BINARY_DIR}/network-contract-data")
        set_tests_properties(lffw_network_contract PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-002;SP-LFFASTESTWHISPER-003;SP-LFFASTESTWHISPER-013;SP-LFFASTESTWHISPER-004;SP-LFFASTESTWHISPER-008;SP-LFFASTESTWHISPER-016;SP-LFFASTESTWHISPER-031;SP-LFFASTESTWHISPER-033")
        add_executable(lffw_network_scratch tests/network_scratch.cpp)
        target_include_directories(lffw_network_scratch SYSTEM PRIVATE "${LFFW_NETWORK_CUDA_INCLUDE}")
        target_link_libraries(lffw_network_scratch PRIVATE lffw_network tensoracle::tensoracle TensorRT::nvinfer)
        add_test(NAME lffw_network_scratch COMMAND lffw_network_scratch
            "${CMAKE_CURRENT_BINARY_DIR}/network-scratch-data")
        set_tests_properties(lffw_network_scratch PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-037;SP-LFFASTESTWHISPER-033")
    endif()
endif()
if(LFFW_BUILD_RUNTIME)
    add_library(lffw_runtime SHARED src/whisper_runtime.cpp)
    target_compile_features(lffw_runtime PUBLIC cxx_std_17)
    target_include_directories(lffw_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(lffw_runtime PRIVATE lffw_network tensoracle::tensoracle TensorRT::nvinfer tensoracle_cuda_driver
        Threads::Threads CUDA::cudart_static)
    # Build-tree C/C++ consumers resolve the private shared SDK dependencies
    # without putting their absolute directories in the installed RUNPATH.
    target_link_options(lffw_runtime INTERFACE
        "LINKER:-rpath-link,${LFFW_TENSORACLE_LIBRARY_DIR}"
        "LINKER:-rpath-link,${LFFW_TENSORRT_ROOT}/lib")
    set_target_properties(lffw_runtime PROPERTIES
        POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
        BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN/../../tensoracle/cpp/lib")
    set(lffw_runtime_map "${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_runtime.map")
    if(LFFW_RUNTIME_DIAGNOSTICS)
        target_sources(lffw_runtime PRIVATE src/runtime_diagnostics.cpp)
        target_compile_definitions(lffw_runtime PRIVATE LFFW_RUNTIME_DIAGNOSTICS=1)
        file(READ "${lffw_runtime_map}" lffw_runtime_exports)
        set(lffw_diagnostic_exports)
        foreach(name IN ITEMS create begin end get_counts get_operation get_span get_sleep destroy)
            string(APPEND lffw_diagnostic_exports "        lffw_diagnostic_${name};\n")
        endforeach()
        string(REPLACE "    local:" "${lffw_diagnostic_exports}    local:"
            lffw_runtime_exports "${lffw_runtime_exports}")
        set(lffw_runtime_map "${CMAKE_CURRENT_BINARY_DIR}/whisper_runtime_diagnostics.map")
        file(WRITE "${lffw_runtime_map}" "${lffw_runtime_exports}")
    endif()
    target_link_options(lffw_runtime PRIVATE
        "LINKER:--version-script=${lffw_runtime_map}"
        "LINKER:--exclude-libs,ALL")
    set_property(TARGET lffw_runtime APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_runtime.map")
    if(BUILD_TESTING)
        lffw_test_static_cuda_runtime(lffw_runtime)
        if(LFFW_RUNTIME_DIAGNOSTICS)
            add_library(lffw_runtime_diagnostic_observer SHARED tests/runtime_fault_observer.cpp)
            target_include_directories(lffw_runtime_diagnostic_observer PRIVATE "${LFFW_NETWORK_CUDA_INCLUDE}")
            target_link_libraries(lffw_runtime_diagnostic_observer PRIVATE tensoracle::tensoracle TensorRT::nvinfer Threads::Threads)
            add_executable(lffw_runtime_diagnostic_observation tests/runtime_diagnostic_observer.cpp)
            target_compile_options(lffw_runtime_diagnostic_observation PRIVATE -UNDEBUG)
            target_link_libraries(lffw_runtime_diagnostic_observation PRIVATE lffw_runtime_diagnostic_observer lffw_runtime)
            add_test(NAME lffw_runtime_diagnostic_observation COMMAND lffw_runtime_diagnostic_observation)
            set_tests_properties(lffw_runtime_diagnostic_observation PROPERTIES
                LABELS "SP-LFFASTESTWHISPER-001;SP-LFFASTESTWHISPER-017")
            add_executable(lffw_runtime_diagnostic_consumer tests/runtime_diagnostics.cpp)
            target_compile_options(lffw_runtime_diagnostic_consumer PRIVATE -UNDEBUG)
            target_link_libraries(lffw_runtime_diagnostic_consumer PRIVATE lffw_runtime Threads::Threads)
            add_test(NAME lffw_runtime_diagnostic_consumer COMMAND lffw_runtime_diagnostic_consumer)
            set_tests_properties(lffw_runtime_diagnostic_consumer PROPERTIES
                LABELS "SP-LFFASTESTWHISPER-001;SP-LFFASTESTWHISPER-017")
        endif()
        foreach(kind IN ITEMS c cpp)
            add_executable(lffw_runtime_${kind}_consumer tests/runtime_${kind}_consumer.${kind})
            target_link_libraries(lffw_runtime_${kind}_consumer PRIVATE lffw_runtime)
            add_test(NAME lffw_runtime_${kind}_consumer COMMAND lffw_runtime_${kind}_consumer)
            set_tests_properties(lffw_runtime_${kind}_consumer PROPERTIES LABELS "SP-LFFASTESTWHISPER-031;SP-LFFASTESTWHISPER-033")
        endforeach()
        foreach(kind IN ITEMS wait buffers cache)
            add_executable(lffw_runtime_${kind} tests/runtime_${kind}.cpp)
            target_compile_features(lffw_runtime_${kind} PRIVATE cxx_std_17)
            target_include_directories(lffw_runtime_${kind} PRIVATE src include "${LFFW_NETWORK_CUDA_INCLUDE}")
            add_test(NAME lffw_runtime_${kind} COMMAND lffw_runtime_${kind})
            set_tests_properties(lffw_runtime_${kind} PROPERTIES LABELS "SP-LFFASTESTWHISPER-001;SP-LFFASTESTWHISPER-017")
        endforeach()
        find_program(LFFW_RUNTIME_NM nm REQUIRED)
        add_test(NAME lffw_runtime_exported_c_functions
            COMMAND "${CMAKE_COMMAND}" "-DNM=${LFFW_RUNTIME_NM}"
                "-DDIAGNOSTICS=${LFFW_RUNTIME_DIAGNOSTICS}"
                "-DLIBRARY=$<TARGET_FILE:lffw_runtime>" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_runtime_symbols.cmake")
        set_tests_properties(lffw_runtime_exported_c_functions PROPERTIES LABELS "SP-LFFASTESTWHISPER-033")
    endif()
endif()
function(lffw_test_sdk_environment)
    get_property(tests DIRECTORY PROPERTY TESTS)
    foreach(test IN LISTS tests)
        set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${LFFW_TEST_TENSORRT_LIBRARY_DIR}")
        set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${LFFW_TENSORACLE_LIBRARY_DIR}")
    endforeach()
endfunction()
if(NOT LFFW_BUILD_PLUGINS)
    lffw_test_sdk_environment()
    return()
endif()

foreach(root IN ITEMS LFFW_BUNDLE_ROOT)
    if(NOT DEFINED ${root} OR "${${root}}" STREQUAL "")
        message(FATAL_ERROR "Set ${root} to the verified build input")
    endif()
endforeach()
find_package(Threads REQUIRED)
tensoracle_add_cubin_plugin(
    TARGET lffw_fc1_plugin
    TENSORRT_ROOT "${LFFW_TENSORRT_ROOT}"
    BUNDLE_ROOT "${LFFW_BUNDLE_ROOT}"
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/fc1_registration.cpp" "${CMAKE_CURRENT_LIST_DIR}/fc1_plugin.cpp")
target_link_libraries(lffw_fc1_plugin PRIVATE TensorRT::nvinfer Threads::Threads CUDA::cudart_static)
set_target_properties(lffw_fc1_plugin PROPERTIES SKIP_BUILD_RPATH TRUE)

include(CTest)
if(BUILD_TESTING)
    lffw_test_static_cuda_runtime(lffw_fc1_plugin)
    tensoracle_add_cubin_plugin(
        TARGET lffw_cubin_plugin
        TENSORRT_ROOT "${LFFW_TENSORRT_ROOT}"
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

    add_executable(lffw_plugin_contracts tests/plugin_contracts.cpp)
    target_compile_features(lffw_plugin_contracts PRIVATE cxx_std_17)
    target_include_directories(lffw_plugin_contracts PRIVATE "${LFFW_TENSORRT_ROOT}/include")
    target_link_libraries(lffw_plugin_contracts PRIVATE tensoracle_cuda_driver ${CMAKE_DL_LIBS})
    # Plugin-only loading exercises creator and field parsing without CUDA initialization.
    set_target_properties(lffw_plugin_contracts PROPERTIES BUILD_RPATH "${LFFW_TENSORRT_ROOT}/lib")
    foreach(plugin IN ITEMS lffw_fc1_plugin lffw_cubin_plugin)
        if(plugin STREQUAL "lffw_fc1_plugin")
            set(creator_count 1)
        else()
            set(creator_count 3)
        endif()
        add_test(NAME ${plugin}_creator_fields_and_shapes
            COMMAND lffw_plugin_contracts "$<TARGET_FILE:${plugin}>" "${creator_count}")
        set_tests_properties(${plugin}_creator_fields_and_shapes PROPERTIES
            LABELS "SP-LFFASTESTWHISPER-003;SP-LFFASTESTWHISPER-007;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-033")
    endforeach()

    add_executable(lffw_bundle_contracts tests/bundle_contracts.cpp)
    target_compile_features(lffw_bundle_contracts PRIVATE cxx_std_17)
    target_link_libraries(lffw_bundle_contracts PRIVATE tensoracle::tensoracle TensorRT::nvinfer)
    add_test(NAME lffw_bundle_inspection
        COMMAND lffw_bundle_contracts "${LFFW_BUNDLE_ROOT}"
            "$<TARGET_FILE:lffw_fc1_plugin>" "$<TARGET_FILE:lffw_cubin_plugin>"
            "${CMAKE_CURRENT_BINARY_DIR}/bundle-inspection-scratch")
    set_tests_properties(lffw_bundle_inspection PROPERTIES
        LABELS "SP-LFFASTESTWHISPER-006;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-033")

    add_executable(lffw_kernel_gpu tests/kernel_gpu.cpp)
    target_compile_features(lffw_kernel_gpu PRIVATE cxx_std_17)
    target_compile_options(lffw_kernel_gpu PRIVATE -Wno-deprecated-declarations)
    target_include_directories(lffw_kernel_gpu PRIVATE "${LFFW_TENSORRT_ROOT}/include")
    target_link_libraries(lffw_kernel_gpu PRIVATE tensoracle::tensoracle TensorRT::nvinfer tensoracle_cuda_driver ${CMAKE_DL_LIBS})
    foreach(kernel IN ITEMS bias qkv fc1)
        if(kernel STREQUAL "fc1")
            set(kernel_plugin lffw_fc1_plugin)
        else()
            set(kernel_plugin lffw_cubin_plugin)
        endif()
        add_test(NAME lffw_${kernel}_fp16_values
            COMMAND lffw_kernel_gpu "${kernel}" "$<TARGET_FILE:${kernel_plugin}>")
        set_tests_properties(lffw_${kernel}_fp16_values PROPERTIES
            LABELS "gpu;SP-LFFASTESTWHISPER-003;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-012"
            TIMEOUT 600 RESOURCE_LOCK lffw_gpu)
    endforeach()
    if(LFFW_FC1_TEST_WEIGHTS)
        add_test(NAME lffw_fc1_checkpoint_values
            COMMAND lffw_kernel_gpu fc1 "$<TARGET_FILE:lffw_fc1_plugin>" "${LFFW_FC1_TEST_WEIGHTS}")
        set_tests_properties(lffw_fc1_checkpoint_values PROPERTIES
            LABELS "gpu;SP-LFFASTESTWHISPER-003;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-012"
            TIMEOUT 600 RESOURCE_LOCK lffw_gpu)
    endif()

    # A separate test observer interposes only calls originating in the plugin.
    # It is never linked into the production library.
    add_library(lffw_qkv_driver_observer SHARED tests/qkv_driver_observer.cpp)
    target_compile_features(lffw_qkv_driver_observer PRIVATE cxx_std_17)
    target_link_libraries(lffw_qkv_driver_observer PRIVATE
        tensoracle_cuda_driver Threads::Threads ${CMAKE_DL_LIBS})

    tensoracle_add_cubin_plugin(
        TARGET lffw_qkv_v1_fixture
        TENSORRT_ROOT "${LFFW_TENSORRT_ROOT}"
        BUNDLE_ROOT "${LFFW_BUNDLE_ROOT}"
        SOURCES "${CMAKE_CURRENT_LIST_DIR}/tests/qkv_v1_fixture.cpp")
    target_link_libraries(lffw_qkv_v1_fixture PRIVATE Threads::Threads)
    set_target_properties(lffw_qkv_v1_fixture PROPERTIES SKIP_BUILD_RPATH TRUE)

    # These fixtures own real TensorRT graphs; each scenario supervises fresh
    # child processes and removes its private cache on success or failure.
    set(LFFW_PLUGIN_TEST_TEMP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.nf/tmp/ctest-plugin")
    add_executable(lffw_plugin_engine_runtime tests/plugin_engine_runtime.cpp)
    target_compile_features(lffw_plugin_engine_runtime PRIVATE cxx_std_17)
    target_compile_options(lffw_plugin_engine_runtime PRIVATE -Wno-deprecated-declarations)
    target_include_directories(lffw_plugin_engine_runtime PRIVATE "${LFFW_TENSORRT_ROOT}/include")
    target_link_libraries(lffw_plugin_engine_runtime PRIVATE tensoracle::tensoracle TensorRT::nvinfer
        tensoracle_cuda_driver)
    add_executable(lffw_plugin_test_scenario tests/plugin_test_scenario.cpp)
    target_compile_features(lffw_plugin_test_scenario PRIVATE cxx_std_17)
    target_link_libraries(lffw_plugin_test_scenario PRIVATE Threads::Threads)
    add_test(NAME lffw_plugin_test_cleanup
        COMMAND lffw_plugin_test_scenario --cleanup-self-test "${LFFW_PLUGIN_TEST_TEMP_ROOT}")
    set_tests_properties(lffw_plugin_test_cleanup PROPERTIES LABELS "SP-LFFASTESTWHISPER-033")

    foreach(kernel IN ITEMS bias qkv fc1)
        if(kernel STREQUAL "fc1")
            set(plugin lffw_fc1_plugin)
            set(scenarios data values reuse contexts reject rounding)
        elseif(kernel STREQUAL "qkv")
            set(plugin lffw_cubin_plugin)
            set(scenarios data values reuse contexts reject legacy)
        else()
            set(plugin lffw_cubin_plugin)
            set(scenarios data values reuse)
        endif()
        foreach(scenario IN LISTS scenarios)
            set(test_name lffw_${kernel}_engine_${scenario})
            add_test(NAME ${test_name}
                COMMAND lffw_plugin_test_scenario "$<TARGET_FILE:lffw_plugin_engine_runtime>"
                    "${kernel}" "${scenario}" "${LFFW_BUNDLE_ROOT}" "$<TARGET_FILE:${plugin}>"
                    "$<TARGET_FILE:lffw_qkv_v1_fixture>" "$<TARGET_FILE:lffw_qkv_driver_observer>"
                    "${LFFW_PLUGIN_TEST_TEMP_ROOT}")
            set_tests_properties(${test_name} PROPERTIES
                TIMEOUT 600 RESOURCE_LOCK lffw_gpu
                LABELS "SP-LFFASTESTWHISPER-033")
            if(NOT scenario STREQUAL "data")
                set_property(TEST ${test_name} APPEND PROPERTY LABELS
                    "gpu;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-012")
                if(scenario STREQUAL "reuse" OR scenario STREQUAL "contexts" OR scenario STREQUAL "legacy")
                    set_property(TEST ${test_name} APPEND PROPERTY LABELS "SP-LFFASTESTWHISPER-008")
                endif()
                if(scenario STREQUAL "values" OR scenario STREQUAL "rounding")
                    set_property(TEST ${test_name} APPEND PROPERTY LABELS "SP-LFFASTESTWHISPER-003")
                endif()
            endif()
        endforeach()
    endforeach()
    if(LFFW_FC1_TEST_WEIGHTS)
        add_test(NAME lffw_fc1_checkpoint_engine_values
            COMMAND lffw_plugin_test_scenario "$<TARGET_FILE:lffw_plugin_engine_runtime>" fc1 values
                "${LFFW_BUNDLE_ROOT}" "$<TARGET_FILE:lffw_fc1_plugin>"
                "$<TARGET_FILE:lffw_qkv_v1_fixture>" "$<TARGET_FILE:lffw_qkv_driver_observer>"
                "${LFFW_PLUGIN_TEST_TEMP_ROOT}" "${LFFW_FC1_TEST_WEIGHTS}")
        set_tests_properties(lffw_fc1_checkpoint_engine_values PROPERTIES
            TIMEOUT 600 RESOURCE_LOCK lffw_gpu
            LABELS "gpu;SP-LFFASTESTWHISPER-003;SP-LFFASTESTWHISPER-011;SP-LFFASTESTWHISPER-012")
    endif()
endif()

if(BUILD_TESTING)
    lffw_test_sdk_environment()
endif()
