cmake_minimum_required(VERSION 3.24)
project(twarz)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Wskazujemy stabilną wersję 4 zainstalowaną przez Homebrew
set(OpenCV_DIR "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

add_executable(twarz main.cpp
)

target_include_directories(twarz PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(twarz PRIVATE ${OpenCV_LIBS})
add_executable(rysy_twarzy main.cpp)
