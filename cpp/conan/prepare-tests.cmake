cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/bootstrap.cmake")
set(LFFW_CONAN_PROFILE "${CMAKE_CURRENT_LIST_DIR}/profiles/linux-gcc-toolset-15"
    CACHE FILEPATH "Conan host and build profile")
get_filename_component(lffw_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
lffw_install_conan("${LFFW_CONAN_PROFILE}" "${lffw_root}/build/conan-prepare" FALSE)
