cmake_minimum_required(VERSION 3.0.0)
project(tf VERSION 0.1.0)

include_directories(include)

set(TARGET tf)
SET(CMAKE_CXX_FLAGS "-larmadillo")

file(GLOB SOURCES "src/*.cpp")

add_executable(${TARGET} ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
