cmake_minimum_required(VERSION 3.10)
project(VideoBackgroundSubtraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(VideoBackgroundSubtraction src/main.cpp )

target_link_libraries(VideoBackgroundSubtraction ${OpenCV_LIBS})
