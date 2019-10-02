cmake_minimum_required(VERSION 3.0.0)
project(medical-cnn VERSION 0.1.0)

include_directories(include)
include_directories(src/layers)
include_directories(src/model_zoo)

if (UNIX)
    set(CMAKE_CXX_FLAGS "-O2 -larmadillo")
else()
    set (ARMADILLO_INCLUDE_DIR "C:/Users/mxw/Downloads/armadillo-9.700.2/include")
    set (ARMADILLO_LIBRARY "C:/Users/mxw/Downloads/armadillo-9.700.2/build/libarmadillo.dll.a")

endif()

find_package(Armadillo REQUIRED)
#find_package(OpenCV REQUIRED )

include_directories(${ARMADILLO_INCLUDE_DIRS})

set (CMAKE_CXX_STANDARD 17)

file(GLOB SOURCES "src/*.cpp" "src/layers/*.cpp" "src/model_zoo/*.cpp")

add_executable(medical-cnn ${SOURCES})

target_link_libraries(medical-cnn ${ARMADILLO_LIBRARIES})
#target_link_libraries(medical-cnn ${OpenCV_LIBS})
