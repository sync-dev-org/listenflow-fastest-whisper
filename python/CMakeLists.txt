cmake_minimum_required(VERSION 3.26)
# A source archive carries cpp/ here; a checkout keeps it one directory above.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
    set(lffw_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
else()
    set(lffw_cpp "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
endif()
include("${lffw_cpp}/conan/bootstrap.cmake")
project(lffw_python LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
if(NOT Python_INTERPRETER_ID STREQUAL "Python" OR NOT TARGET Python::SABIModule)
    message(FATAL_ERROR "The model extension requires GIL-enabled CPython 3.12 or newer")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(int(bool(sysconfig.get_config_var('Py_GIL_DISABLED'))))"
    OUTPUT_VARIABLE lffw_free_threaded OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(lffw_free_threaded)
    message(FATAL_ERROR "The model extension requires GIL-enabled CPython")
endif()
# Nanobind is supplied by the Python build-system requirements.
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind 2.15.0 EXACT CONFIG REQUIRED)

# Both entry points compile the standalone production target.
add_subdirectory("${lffw_cpp}" cpp-build)
nanobind_add_module(_native STABLE_ABI "${lffw_cpp}/bindings.cpp")
target_link_libraries(_native PRIVATE lffw_runtime)
if(LFFW_RUNTIME_DIAGNOSTICS)
    nanobind_add_module(_lffw_diagnostics STABLE_ABI "${lffw_cpp}/diagnostic_bindings.cpp")
    target_link_libraries(_lffw_diagnostics PRIVATE lffw_runtime)
    set_target_properties(_lffw_diagnostics PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_test(NAME lffw_product_module_has_no_diagnostics
        COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.util; s=importlib.util.spec_from_file_location('_native',r'$<TARGET_FILE:_native>'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'Session' not in vars(m) and not any('diagnostic' in k.lower() for k in vars(m))")
    set_tests_properties(lffw_product_module_has_no_diagnostics PROPERTIES LABELS "SP-LFFASTESTWHISPER-033")
    set_tests_properties(lffw_product_module_has_no_diagnostics PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${LFFW_TENSORACLE_LIBRARY_DIR}:${LFFW_TEST_TENSORRT_LIBRARY_DIR}")
endif()
set_target_properties(_native PROPERTIES BUILD_RPATH_USE_ORIGIN ON INSTALL_RPATH "$ORIGIN")
install(TARGETS _native lffw_runtime LIBRARY DESTINATION listenflow/fastest_whisper)
install(TARGETS lffw_fc1_plugin LIBRARY DESTINATION listenflow/fastest_whisper)
