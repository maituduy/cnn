cmake_minimum_required(VERSION 3.0.0)
project(medical-cnn VERSION 0.1.0)

include_directories(include)
find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})

# set(TARGET "bin/output")
set(CMAKE_CXX_FLAGS "-larmadillo")

file(GLOB SOURCES "src/*.cpp")

add_executable(medical-cnn ${SOURCES})

target_link_libraries(medical-cnn ${ARMADILLO_LIBRARIES})
