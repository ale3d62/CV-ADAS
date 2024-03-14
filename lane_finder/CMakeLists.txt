cmake_minimum_required(VERSION 3.0.0)
project(lane_finder)


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} video_test.cpp)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} winmm.lib)