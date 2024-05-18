cmake_minimum_required(VERSION 3.28)
project(AtypicalLibrary)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
        glog::glog
        pthread
)