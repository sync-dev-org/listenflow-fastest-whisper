cmake_minimum_required(VERSION 3.20)
project(lffw_python LANGUAGES CXX)

# The source archive carries cpp/ alongside this file; a checkout keeps it one
# directory above. Both builds compile the standalone production target.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
    set(lffw_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
else()
    set(lffw_cpp "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
endif()
add_subdirectory("${lffw_cpp}" cpp-build)
install(TARGETS lffw_fc1_plugin LIBRARY DESTINATION listenflow/fastest_whisper)
