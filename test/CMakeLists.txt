cmake_minimum_required(VERSION 3.22)
project(ATL_test)
set(CMAKE_CXX_STANDARD 20)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "Torch LIB path: " ${TORCH_LIBRARIES})
message(STATUS "Torch include path: " ${TORCH_INCLUDE_DIRS})

file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

include_directories(
        ${TORCH_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})

target_link_libraries(${PROJECT_NAME}
        AtypicalLibrary
        GTest::gtest_main
        glog::glog
        /opt/anaconda3/envs/dl/lib/python3.8/site-packages/torch/lib/libc10.dylib
#        ${TORCH_LIBRARIES}
#        /home/richard/anaconda3/envs/dl/lib/python3.12/site-packages/torch/lib/libc10.so
#        /home/richard/anaconda3/envs/dl/lib/python3.12/site-packages/torch/lib/libtorch.so
#        /home/richard/anaconda3/envs/dl/lib/python3.12/site-packages/torch/lib/libtorch_global_deps.so
)
